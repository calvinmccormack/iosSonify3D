import SwiftUI
import ARKit

private let sortedCocoClasses: [(name: String, id: Int)] = {
    cocoClasses.sorted { $0.key < $1.key }.map { (name: $0.key, id: $0.value) }
}()

struct ContentView: View {
    @StateObject private var depthPipeline = DepthPipeline()
    @StateObject private var audio = SpectralAudioEngine()
    @StateObject private var voiceManager = VoiceCommandManager()

    @State private var sweepSeconds: Double = 2.0
    @State private var confidence: Double = 0.25

    // 4 target slots, -1 = none
    @State private var slotSelections: [Int] = [-1, -1, -1, -1]
    @State private var showingPickerFor: Int? = nil

    // Slot colors for visual distinction
    private let slotColors: [Color] = [.red, .blue, .green, .orange]
    private let slotLabels = ["① Brass", "② Hollow", "③ Bell", "④ Nasal"]

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 12) {
                // LEFT: Preview
                ZStack(alignment: .topLeading) {
                    if let image = depthPipeline.debugImage {
                        Image(uiImage: image)
                            .resizable()
                            .interpolation(.none)
                            .scaledToFit()
                            .aspectRatio(4.0/3.0, contentMode: .fit)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .border(Color.gray)
                    } else {
                        Text("Waiting for depth…")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .border(Color.gray)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        if !depthPipeline.detectionStatusText.isEmpty {
                            Text(depthPipeline.detectionStatusText)
                                .font(.caption)
                                .padding(4)
                                .background(Color.black.opacity(0.7))
                                .foregroundColor(.white)
                                .cornerRadius(4)
                        }
                        if depthPipeline.isScanning {
                            Text("Scanning… col \(depthPipeline.scanColumn + 1)/\(DepthPipeline.gridWidth)")
                                .font(.caption2).bold()
                                .padding(4)
                                .background(Color.red.opacity(0.8))
                                .foregroundColor(.white)
                                .cornerRadius(4)
                        }
                    }
                    .padding(8)
                }
                .frame(width: geo.size.width * 0.50, height: geo.size.height)
                .clipped()

                // RIGHT: Controls
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 6) {

                        // Scan + Mic
                        HStack {
                            Button {
                                if depthPipeline.isScanning { cancelScan() }
                                else { triggerScan() }
                            } label: {
                                HStack(spacing: 5) {
                                    Image(systemName: depthPipeline.isScanning
                                          ? "stop.fill" : "waveform.circle.fill")
                                    Text(depthPipeline.isScanning ? "Stop" : "Scan")
                                }
                                .font(.callout)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(depthPipeline.isScanning ? .red : .blue)

                            Spacer()

                            Button {
                                if voiceManager.isListening { voiceManager.stopListening() }
                                else { voiceManager.startListening() }
                            } label: {
                                Image(systemName: voiceManager.isListening
                                      ? "mic.fill" : "mic.slash")
                                    .font(.title3)
                                    .foregroundColor(voiceManager.isListening ? .red : .secondary)
                            }
                        }

                        if voiceManager.isListening {
                            Text(voiceManager.statusText)
                                .font(.caption2).foregroundColor(.secondary)
                        }

                        Divider()

                        // 4 target slots
                        Text("Targets").font(.caption).bold()

                        ForEach(0..<4, id: \.self) { slot in
                            Button { showingPickerFor = slot } label: {
                                HStack(spacing: 6) {
                                    Circle().fill(slotColors[slot])
                                        .frame(width: 8, height: 8)
                                    if slotSelections[slot] >= 0 {
                                        Text("\(slot+1): \(className(for: slotSelections[slot]))")
                                    } else {
                                        Text("\(slotLabels[slot]): —")
                                            .foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                        .foregroundColor(.secondary)
                                        .font(.caption2)
                                }
                                .font(.caption)
                                .padding(5)
                                .background(Color(.systemGray6))
                                .cornerRadius(5)
                            }
                            .buttonStyle(.plain)
                        }

                        Divider()

                        LabeledSlider(title: "Conf", value: $confidence,
                                      range: 0.05...0.95, format: "%.2f")
                            .onChange(of: confidence) { _, v in
                                depthPipeline.detector.confidenceThreshold = Float(v)
                            }

                        LabeledSlider(title: "Sweep", value: $sweepSeconds,
                                      range: 0.5...8.0, format: "%.1fs")

                        Spacer(minLength: 2)

                        Text(String(format: "%.0f FPS", depthPipeline.fps))
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }
                .frame(width: geo.size.width * 0.50 - 24, height: geo.size.height)
            }
            .padding(12)
        }
        .overlay(
            ARDepthCaptureView(pipeline: depthPipeline).frame(width: 0, height: 0)
        )
        .onAppear {
            depthPipeline.detector.confidenceThreshold = Float(confidence)
            audio.start()
            setupVoiceCallbacks()
            voiceManager.requestPermissions { _ in }
        }
        .sheet(item: $showingPickerFor) { slot in
            ClassPickerSheet(selection: Binding(
                get: { slotSelections[slot] },
                set: { slotSelections[slot] = $0 }
            ), title: slotLabels[slot]) {
                updateActiveClasses()
            }
        }
    }

    // MARK: - Scan

    private func triggerScan() {
        guard !depthPipeline.isScanning else { return }
        audio.beginScan()

        depthPipeline.triggerScan(sweepSeconds: sweepSeconds,
            onColumn: { col, hits, z01, pan in
                if hits.isEmpty {
                    self.audio.clearColumn(pan: pan)
                } else {
                    for hit in hits {
                        self.audio.setTarget(slot: hit.slot, centroid: hit.centroid,
                                             coverage: hit.coverage, depth: z01, pan: pan)
                    }
                }
            },
            onComplete: {
                self.audio.endScan()
            }
        )
    }

    private func cancelScan() {
        depthPipeline.stopScan()
        audio.endScan()
    }

    // MARK: - Slot management

    private func updateActiveClasses() {
        // Build classId → slot map
        var slotMap: [Int: Int] = [:]
        for slot in 0..<4 {
            let cid = slotSelections[slot]
            if cid >= 0 { slotMap[cid] = slot }
        }
        depthPipeline.setActiveClasses(slotMap)

        // Configure audio voice slots
        for slot in 0..<4 {
            let cid = slotSelections[slot]
            if cid >= 0 {
                audio.assignSlot(slot, classId: cid)
            }
        }
        if slotMap.isEmpty { audio.deactivateAll() }
    }

    private func setupVoiceCallbacks() {
        voiceManager.onSlotChanged = { slot, classId in
            guard slot >= 0 && slot < 4 else { return }
            DispatchQueue.main.async {
                self.slotSelections[slot] = classId
                self.updateActiveClasses()
            }
        }
        voiceManager.onClearAll = {
            DispatchQueue.main.async {
                self.slotSelections = [-1, -1, -1, -1]
                self.updateActiveClasses()
            }
        }
        voiceManager.onSweepRateChanged = { multiplier in
            self.sweepSeconds = 2.0 * Double(multiplier)
        }
        voiceManager.onScanTriggered = {
            DispatchQueue.main.async { self.triggerScan() }
        }
    }

    private func className(for id: Int) -> String {
        cocoClasses.first(where: { $0.value == id })?.key.capitalized ?? "Unknown"
    }
}

// Make Int identifiable for .sheet(item:)
extension Int: @retroactive Identifiable {
    public var id: Int { self }
}

// MARK: - Class Picker Sheet

struct ClassPickerSheet: View {
    @Binding var selection: Int
    let title: String
    var onDismiss: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""

    private var filteredClasses: [(name: String, id: Int)] {
        let all = sortedCocoClasses
        if searchText.isEmpty { return all }
        return all.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        NavigationView {
            List {
                Button {
                    selection = -1; onDismiss(); dismiss()
                } label: {
                    HStack {
                        Text("None").foregroundColor(.primary)
                        Spacer()
                        if selection == -1 {
                            Image(systemName: "checkmark").foregroundColor(.blue)
                        }
                    }
                }

                ForEach(filteredClasses, id: \.id) { cls in
                    Button {
                        selection = cls.id; onDismiss(); dismiss()
                    } label: {
                        HStack {
                            Text(cls.name.capitalized).foregroundColor(.primary)
                            Spacer()
                            if selection == cls.id {
                                Image(systemName: "checkmark").foregroundColor(.blue)
                            }
                        }
                    }
                }
            }
            .searchable(text: $searchText, prompt: "Search objects")
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Slider

private struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String

    var body: some View {
        HStack {
            Text(title).font(.caption).frame(width: 42, alignment: .leading)
            Slider(value: $value, in: range)
            Text(String(format: format, value))
                .font(.caption2).monospacedDigit()
                .frame(width: 46, alignment: .trailing)
        }
    }
}
