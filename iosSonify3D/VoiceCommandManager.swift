import Foundation
import Speech
import AVFoundation
import Combine

/// Maps COCO class names to YOLOv8 class IDs (0-79)
let cocoClasses: [String: Int] = [
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33, "baseball bat": 34,
    "baseball glove": 35, "skateboard": 36, "surfboard": 37, "tennis racket": 38,
    "bottle": 39, "wine glass": 40, "cup": 41, "fork": 42, "knife": 43,
    "spoon": 44, "bowl": 45, "banana": 46, "apple": 47, "sandwich": 48,
    "orange": 49, "broccoli": 50, "carrot": 51, "hot dog": 52, "pizza": 53,
    "donut": 54, "cake": 55, "chair": 56, "couch": 57, "potted plant": 58,
    "bed": 59, "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63,
    "mouse": 64, "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77, "hair drier": 78,
    "toothbrush": 79
]

private let classAliases: [String: String] = [
    "phone": "cell phone", "cellphone": "cell phone", "mobile": "cell phone",
    "sofa": "couch", "television": "tv", "monitor": "tv",
    "table": "dining table", "desk": "dining table",
    "glass": "wine glass", "mug": "cup",
    "bag": "handbag", "purse": "handbag",
    "plant": "potted plant", "flower": "potted plant",
    "fridge": "refrigerator", "ball": "sports ball",
    "bike": "bicycle", "motorbike": "motorcycle"
]

/// Resolve a spoken name to a COCO class ID. Exact match first, then alias, then partial.
func resolveCocoClass(_ input: String) -> (name: String, id: Int)? {
    let cleaned = input.trimmingCharacters(in: .punctuationCharacters).lowercased()
    // 1. Exact match
    if let id = cocoClasses[cleaned] { return (cleaned, id) }
    // 2. Alias match
    if let canonical = classAliases[cleaned], let id = cocoClasses[canonical] { return (canonical, id) }
    // 3. Partial match — only if query is 3+ chars to avoid false positives
    if cleaned.count >= 3 {
        // Prefer exact word containment over substring
        for (className, classId) in cocoClasses {
            if className == cleaned { return (className, classId) }
        }
        for (className, classId) in cocoClasses {
            // Only match if query IS the classname or classname IS the query (full word)
            let classWords = className.split(separator: " ").map { String($0) }
            let queryWords = cleaned.split(separator: " ").map { String($0) }
            for cw in classWords {
                for qw in queryWords {
                    if cw == qw { return (className, classId) }
                }
            }
        }
    }
    return nil
}

final class VoiceCommandManager: NSObject, ObservableObject, SFSpeechRecognizerDelegate {

    @Published var activeClasses: [(name: String, classId: Int)] = []
    @Published var statusText: String = "Tap mic to listen"
    @Published var isListening: Bool = false
    @Published var sweepRateMultiplier: Float = 1.0

    var onClassesChanged: (([Int]) -> Void)?
    var onSweepRateChanged: ((Float) -> Void)?

    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()

    private var executedCommands: Set<String> = []
    static let maxActiveClasses = 2

    override init() {
        super.init()
        speechRecognizer?.delegate = self
    }

    func requestPermissions(completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                completion(status == .authorized)
                if status != .authorized { self.statusText = "Speech not authorized" }
            }
        }
    }

    func startListening() {
        guard !isListening else { return }
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            statusText = "Recognizer unavailable"; return
        }

        recognitionTask?.cancel()
        recognitionTask = nil
        executedCommands.removeAll()

        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default,
                                         options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("[Voice] Audio session error: \(error)"); statusText = "Audio error"; return
        }

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let request = recognitionRequest else { return }
        request.shouldReportPartialResults = true
        if #available(iOS 13, *) {
            request.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
        }

        let inputNode = audioEngine.inputNode
        inputNode.removeTap(onBus: 0)
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            self.recognitionRequest?.append(buffer)
        }

        audioEngine.prepare()
        do {
            try audioEngine.start()
            print("[Voice] Listening started")
        } catch {
            print("[Voice] Engine error: \(error)"); statusText = "Engine error"; return
        }

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }

            if let result = result {
                let fullText = result.bestTranscription.formattedString.lowercased()
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                print("[Voice] Transcript: '\(fullText)'")
                self.processFullTranscript(fullText)
            }

            if error != nil || (result?.isFinal ?? false) {
                self.audioEngine.inputNode.removeTap(onBus: 0)
                self.audioEngine.stop()
                self.recognitionRequest = nil
                self.recognitionTask = nil
                if self.isListening {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        if self.isListening { self.startListening() }
                    }
                }
            }
        }

        isListening = true
        statusText = "Listening…"
    }

    func stopListening() {
        isListening = false
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionRequest = nil
        recognitionTask = nil
        statusText = "Tap mic to listen"
        executedCommands.removeAll()
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [])
    }

    private func processFullTranscript(_ text: String) {
        guard !text.isEmpty else { return }

        // "clear" / "reset"
        if (text.contains("clear") || text.contains("reset")) && !executedCommands.contains("clear") {
            executedCommands.insert("clear")
            activeClasses.removeAll()
            notifyClassChange()
            speak("Cleared"); statusText = "Cleared"
            print("[Voice] → clear"); return
        }

        // "slower"
        if text.contains("slower") && !executedCommands.contains("slower") {
            executedCommands.insert("slower")
            sweepRateMultiplier = min(3.0, sweepRateMultiplier * 1.3)
            onSweepRateChanged?(sweepRateMultiplier)
            speak("Slower"); print("[Voice] → slower"); return
        }

        // "faster"
        if text.contains("faster") && !executedCommands.contains("faster") {
            executedCommands.insert("faster")
            sweepRateMultiplier = max(0.3, sweepRateMultiplier / 1.3)
            onSweepRateChanged?(sweepRateMultiplier)
            speak("Faster"); print("[Voice] → faster"); return
        }

        // "target 2 [object]" or "target two [object]"
        let t2Patterns = ["target 2 ", "target two ", "target to "]
        for pat in t2Patterns {
            if let range = text.range(of: pat) {
                let after = String(text[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                let objName = extractObjectName(after)
                let cmdKey = "target2:\(objName)"
                if !objName.isEmpty && !executedCommands.contains(cmdKey) {
                    if let resolved = resolveCocoClass(objName) {
                        executedCommands.insert(cmdKey)
                        // Add as second target
                        if activeClasses.count < 2 {
                            activeClasses.append((name: resolved.name, classId: resolved.id))
                        } else {
                            activeClasses[1] = (name: resolved.name, classId: resolved.id)
                        }
                        notifyClassChange()
                        speak("Target 2 \(resolved.name)")
                        statusText = "Finding: " + activeClasses.map { $0.name }.joined(separator: " + ")
                        print("[Voice] → target 2 \(resolved.name) (id \(resolved.id))")
                    }
                }
                return
            }
        }

        // "find [object]"
        if let range = text.range(of: "find ") {
            let after = String(text[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            let objName = extractObjectName(after)
            let cmdKey = "find:\(objName)"
            if !objName.isEmpty && !executedCommands.contains(cmdKey) {
                if let resolved = resolveCocoClass(objName) {
                    executedCommands.insert(cmdKey)
                    setAsTarget(slot: 0, name: resolved.name, classId: resolved.id)
                    speak("Finding \(resolved.name)")
                    print("[Voice] → find \(resolved.name) (id \(resolved.id))")
                }
            }
            return
        }

        // "also [object]"
        if let range = text.range(of: "also ") {
            let after = String(text[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            let objName = extractObjectName(after)
            let cmdKey = "also:\(objName)"
            if !objName.isEmpty && !executedCommands.contains(cmdKey) {
                if let resolved = resolveCocoClass(objName) {
                    executedCommands.insert(cmdKey)
                    if activeClasses.count < Self.maxActiveClasses &&
                       !activeClasses.contains(where: { $0.classId == resolved.id }) {
                        activeClasses.append((name: resolved.name, classId: resolved.id))
                    } else if activeClasses.count >= Self.maxActiveClasses {
                        activeClasses[1] = (name: resolved.name, classId: resolved.id)
                    }
                    notifyClassChange()
                    speak("Also \(resolved.name)")
                    print("[Voice] → also \(resolved.name) (id \(resolved.id))")
                }
            }
            return
        }

        // Bare object name (no "find"/"also" prefix) — treat as "find"
        let words = text.split(separator: " ").map { String($0) }
        if let lastWord = words.last {
            let cmdKey = "bare:\(lastWord)"
            if !executedCommands.contains(cmdKey) {
                if let resolved = resolveCocoClass(lastWord) {
                    executedCommands.insert(cmdKey)
                    setAsTarget(slot: 0, name: resolved.name, classId: resolved.id)
                    speak("Finding \(resolved.name)")
                    print("[Voice] → bare word → find \(resolved.name) (id \(resolved.id))")
                }
            }
        }
    }

    private func setAsTarget(slot: Int, name: String, classId: Int) {
        if activeClasses.isEmpty {
            activeClasses.append((name: name, classId: classId))
        } else {
            activeClasses[min(slot, activeClasses.count - 1)] = (name: name, classId: classId)
        }
        notifyClassChange()
        statusText = "Finding: " + activeClasses.map { $0.name }.joined(separator: " + ")
    }

    private func extractObjectName(_ text: String) -> String {
        let words = text.split(separator: " ").map { String($0) }
        guard !words.isEmpty else { return "" }
        if words.count >= 2 {
            let twoWord = "\(words[0]) \(words[1])"
            if resolveCocoClass(twoWord) != nil { return twoWord }
        }
        return words[0]
    }

    private func notifyClassChange() {
        let ids = activeClasses.map { $0.classId }
        onClassesChanged?(ids)
    }

    private func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.2
        utterance.volume = 0.6
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        synthesizer.speak(utterance)
    }

    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if !available { statusText = "Recognizer unavailable" }
    }
}
