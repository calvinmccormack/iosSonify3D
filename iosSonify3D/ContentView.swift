import SwiftUI
import ARKit

/// Sorted COCO class list for display
private let sortedCocoClasses: [(name: String, id: Int)] = {
    cocoClasses.sorted { $0.key < $1.key }.map { (name: $0.key, id: $0.value) }
}()

struct ContentView: View {
    @StateObject private var depthPipeline = DepthPipeline()
    @StateObject private var audio = SpectralAudioEngine()
    @StateObject private var voiceManager = VoiceCommandManager()

    @State private var isRunning = false
    @State private var sweepSeconds: Double = 2.0
    @State private var fMin: Double = 50
    @State private var fMax: Double = 10050
    @State private var gainRangeDB: Double = 24
    @State private var confidence: Double = 0.25

    // Class selections: -1 = none
    @State private var class1Selection: Int = -1
    @State private var class2Selection: Int = -1

    // Sheet presentation for class selection
    @State private var showingPicker1 = false
    @State private var showingPicker2 = false

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

                    if !depthPipeline.detectionStatusText.isEmpty {
                        Text(depthPipeline.detectionStatusText)
                            .font(.caption)
                            .padding(4)
                            .background(Color.black.opacity(0.7))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                            .padding(8)
                    }
                }
                .frame(width: geo.size.width * 0.55, height: geo.size.height)
                .clipped()

                // RIGHT: Controls
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 8) {

                        // Start/Stop + Mic
                        HStack {
                            Button(isRunning ? "Stop" : "Start") {
                                if isRunning { stop() } else { start() }
                            }
                            .buttonStyle(.borderedProminent)

                            Spacer()

                            Button {
                                if voiceManager.isListening {
                                    voiceManager.stopListening()
                                } else {
                                    voiceManager.startListening()
                                }
                            } label: {
                                Image(systemName: voiceManager.isListening ? "mic.fill" : "mic.slash")
                                    .font(.title2)
                                    .foregroundColor(voiceManager.isListening ? .red : .secondary)
                            }
                        }

                        if voiceManager.isListening {
                            Text(voiceManager.statusText)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }

                        Divider()

                        // Class selection buttons (open sheets)
                        Text("Target Objects").font(.caption).bold()

                        // Target 1
                        Button {
                            showingPicker1 = true
                        } label: {
                            HStack {
                                if class1Selection >= 0 {
                                    Circle().fill(colorForClass(class1Selection)).frame(width: 10, height: 10)
                                    Text(className(for: class1Selection))
                                } else {
                                    Text("Target 1: None")
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Image(systemName: "chevron.right").foregroundColor(.secondary)
                            }
                            .font(.caption)
                            .padding(6)
                            .background(Color(.systemGray6))
                            .cornerRadius(6)
                        }
                        .buttonStyle(.plain)

                        // Target 2
                        Button {
                            showingPicker2 = true
                        } label: {
                            HStack {
                                if class2Selection >= 0 {
                                    Circle().fill(colorForClass(class2Selection)).frame(width: 10, height: 10)
                                    Text(className(for: class2Selection))
                                } else {
                                    Text("Target 2: None")
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Image(systemName: "chevron.right").foregroundColor(.secondary)
                            }
                            .font(.caption)
                            .padding(6)
                            .background(Color(.systemGray6))
                            .cornerRadius(6)
                        }
                        .buttonStyle(.plain)

                        Divider()

                        // Confidence + sliders
                        LabeledSlider(title: "Conf", value: $confidence, range: 0.05...0.95, format: "%.2f")
                            .onChange(of: confidence) { _, v in
                                depthPipeline.detector.confidenceThreshold = Float(v)
                            }

                        LabeledSlider(title: "Sweep", value: $sweepSeconds, range: 0.5...8.0, format: "%.1fs")
                        LabeledSlider(title: "Lo Hz", value: $fMin, range: 20...500, format: "%.0f")
                        LabeledSlider(title: "Hi Hz", value: $fMax, range: 2000...20000, format: "%.0f")
                        LabeledSlider(title: "Atten", value: $gainRangeDB, range: 6...48, format: "%.0fdB")

                        Spacer(minLength: 4)

                        // Status
                        let colText = "Col \(depthPipeline.scanColumn + 1)/\(DepthPipeline.gridWidth)"
                        let fpsText = String(format: "%.0f FPS", depthPipeline.fps)
                        Text("\(colText) • \(fpsText)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(width: geo.size.width * 0.45 - 24, height: geo.size.height)
            }
            .padding(12)
        }
        .overlay(
            ARDepthCaptureView(pipeline: depthPipeline).frame(width: 0, height: 0)
        )
        .onAppear {
            audio.configureBands(fMin: fMin, fMax: fMax)
            depthPipeline.detector.confidenceThreshold = Float(confidence)
            setupVoiceCallbacks()
            voiceManager.requestPermissions { _ in }
        }
        .onChange(of: fMin) { _, v in audio.configureBands(fMin: v, fMax: fMax) }
        .onChange(of: fMax) { _, v in audio.configureBands(fMin: fMin, fMax: v) }
        .onChange(of: gainRangeDB) { _, v in depthPipeline.gainRangeDB = Float(v) }
        .onChange(of: sweepSeconds) { _, v in depthPipeline.setSweepRate(v) }
        .sheet(isPresented: $showingPicker1) {
            ClassPickerSheet(selection: $class1Selection, title: "Target 1") {
                updateActiveClasses()
            }
        }
        .sheet(isPresented: $showingPicker2) {
            ClassPickerSheet(selection: $class2Selection, title: "Target 2") {
                updateActiveClasses()
            }
        }
    }

    private func updateActiveClasses() {
        var ids: [Int] = []
        if class1Selection >= 0 { ids.append(class1Selection) }
        if class2Selection >= 0 && class2Selection != class1Selection { ids.append(class2Selection) }
        depthPipeline.setActiveClasses(ids)
        if ids.isEmpty { audio.deactivateCombs() }
    }

    private func setupVoiceCallbacks() {
        voiceManager.onClassesChanged = { classIds in
            DispatchQueue.main.async {
                self.class1Selection = classIds.count > 0 ? classIds[0] : -1
                self.class2Selection = classIds.count > 1 ? classIds[1] : -1
            }
            self.depthPipeline.setActiveClasses(classIds)
            if classIds.isEmpty { self.audio.deactivateCombs() }
        }
        voiceManager.onSweepRateChanged = { multiplier in
            self.sweepSeconds = 2.0 * Double(multiplier)
            self.depthPipeline.setSweepRate(self.sweepSeconds)
        }
    }

    private func start() {
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else { return }
        audio.start()
        depthPipeline.start(sweepSeconds: sweepSeconds) { col, envelope, targetMask40, classId, pan, z01, edge01 in
            audio.updateEnvelope(envelope)
            if classId >= 0 {
                audio.setTargetBands(targetMask40, classId: classId, boostDB: 12)
            } else {
                audio.clearTarget()
            }
            audio.pan = pan
            audio.updateDistance(z01)
            audio.triggerEdge(edge01)
        }
        isRunning = true
    }

    private func stop() {
        depthPipeline.stop()
        audio.stop()
        isRunning = false
    }

    private func colorForClass(_ classId: Int) -> Color {
        Color(hue: Double(classId * 37 % 360) / 360.0, saturation: 0.8, brightness: 0.9)
    }

    private func className(for id: Int) -> String {
        cocoClasses.first(where: { $0.value == id })?.key.capitalized ?? "Unknown"
    }
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
                // None option
                Button {
                    selection = -1
                    onDismiss()
                    dismiss()
                } label: {
                    HStack {
                        Text("None").foregroundColor(.primary)
                        Spacer()
                        if selection == -1 {
                            Image(systemName: "checkmark").foregroundColor(.blue)
                        }
                    }
                }

                // COCO classes
                ForEach(filteredClasses, id: \.id) { cls in
                    Button {
                        selection = cls.id
                        onDismiss()
                        dismiss()
                    } label: {
                        HStack {
                            Circle()
                                .fill(Color(hue: Double(cls.id * 37 % 360) / 360.0, saturation: 0.8, brightness: 0.9))
                                .frame(width: 10, height: 10)
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
                .font(.caption2)
                .monospacedDigit()
                .frame(width: 46, alignment: .trailing)
        }
    }
}
