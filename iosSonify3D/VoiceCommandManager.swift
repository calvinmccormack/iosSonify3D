import Foundation
import Speech
import AVFoundation
import Combine

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

func resolveCocoClass(_ input: String) -> (name: String, id: Int)? {
    let cleaned = input.trimmingCharacters(in: .punctuationCharacters).lowercased()
    if let id = cocoClasses[cleaned] { return (cleaned, id) }
    if let canonical = classAliases[cleaned], let id = cocoClasses[canonical] {
        return (canonical, id)
    }
    if cleaned.count >= 3 {
        for (className, classId) in cocoClasses {
            if className == cleaned { return (className, classId) }
        }
        for (className, classId) in cocoClasses {
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

    @Published var statusText: String = "Tap mic to listen"
    @Published var isListening: Bool = false
    @Published var sweepRateMultiplier: Float = 1.0

    /// Callbacks — slot-based
    var onSlotChanged: ((Int, Int) -> Void)?    // (slot, classId)
    var onClearAll: (() -> Void)?
    var onSweepRateChanged: ((Float) -> Void)?
    var onScanTriggered: (() -> Void)?

    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()

    private var executedCommands: Set<String> = []

    // Map spoken numbers to slot indices
    private static let numberWords: [(String, Int)] = [
        ("1", 0), ("one", 0), ("won", 0),
        ("2", 1), ("two", 1), ("to", 1), ("too", 1),
        ("3", 2), ("three", 2), ("tree", 2),
        ("4", 3), ("four", 3), ("for", 3), ("fore", 3)
    ]

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

        recognitionTask?.cancel(); recognitionTask = nil
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

        isListening = true; statusText = "Listening…"
    }

    func stopListening() {
        isListening = false
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionRequest = nil; recognitionTask = nil
        statusText = "Tap mic to listen"
        executedCommands.removeAll()
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default, options: [])
    }

    private func processFullTranscript(_ text: String) {
        guard !text.isEmpty else { return }

        // "scan" / "go"
        if (text.hasSuffix("scan") || text.hasSuffix("go") || text == "scan" || text == "go")
            && !executedCommands.contains("scan:\(text.count)") {
            executedCommands.insert("scan:\(text.count)")
            onScanTriggered?()
            speak("Scanning"); statusText = "Scanning…"
            print("[Voice] → scan"); return
        }

        // "clear" / "reset"
        if (text.contains("clear") || text.contains("reset"))
            && !executedCommands.contains("clear") {
            executedCommands.insert("clear")
            onClearAll?()
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

        // "target N [object]" — assign to slot N
        if let range = text.range(of: "target ") {
            let after = String(text[range.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let afterWords = after.split(separator: " ").map { String($0) }
            guard afterWords.count >= 2 else { return }

            // First word should be a number
            let numWord = afterWords[0]
            var slot: Int? = nil
            for (word, idx) in Self.numberWords {
                if numWord == word { slot = idx; break }
            }
            guard let resolvedSlot = slot else {
                // No number — might be "find" style, fall through
                return
            }

            let objText = afterWords.dropFirst().joined(separator: " ")
            let objName = extractObjectName(objText)
            let cmdKey = "target\(resolvedSlot):\(objName)"
            if !objName.isEmpty && !executedCommands.contains(cmdKey) {
                if let resolved = resolveCocoClass(objName) {
                    executedCommands.insert(cmdKey)
                    onSlotChanged?(resolvedSlot, resolved.id)
                    speak("Target \(resolvedSlot + 1) \(resolved.name)")
                    statusText = "Slot \(resolvedSlot + 1): \(resolved.name)"
                    print("[Voice] → target \(resolvedSlot + 1) \(resolved.name) (id \(resolved.id))")
                }
            }
            return
        }

        // "find [object]" → slot 0
        if let range = text.range(of: "find ") {
            let after = String(text[range.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let objName = extractObjectName(after)
            let cmdKey = "find:\(objName)"
            if !objName.isEmpty && !executedCommands.contains(cmdKey) {
                if let resolved = resolveCocoClass(objName) {
                    executedCommands.insert(cmdKey)
                    onSlotChanged?(0, resolved.id)
                    speak("Finding \(resolved.name)")
                    statusText = "Slot 1: \(resolved.name)"
                    print("[Voice] → find \(resolved.name) → slot 0")
                }
            }
            return
        }

        // Bare object name → slot 0
        let words = text.split(separator: " ").map { String($0) }
        if let lastWord = words.last {
            let cmdKey = "bare:\(lastWord)"
            if !executedCommands.contains(cmdKey) {
                if let resolved = resolveCocoClass(lastWord) {
                    executedCommands.insert(cmdKey)
                    onSlotChanged?(0, resolved.id)
                    speak("Finding \(resolved.name)")
                    statusText = "Slot 1: \(resolved.name)"
                    print("[Voice] → bare → \(resolved.name) → slot 0")
                }
            }
        }
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

    private func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.2
        utterance.volume = 0.6
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        synthesizer.speak(utterance)
    }

    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer,
                          availabilityDidChange available: Bool) {
        if !available { statusText = "Recognizer unavailable" }
    }
}
