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

    @State private var sweepSeconds: Double = 2.0
    @State private var confidence: Double = 0.25

    // Class selections: -1 = none
    @State private var class1Selection: Int = -1
    @State private var class2Selection: Int = -1

    // Sheet presentation
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
                .frame(width: geo.size.width * 0.55, height: geo.size.height)
                .clipped()

                // RIGHT: Controls
                ScrollView(.vertical, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 8) {

                        // Scan button + Mic
                        HStack {
                            Button {
                                if depthPipeline.isScanning {
                                    cancelScan()
                                } else {
                                    triggerScan()
                                }
                            } label: {
                                HStack(spacing: 6) {
                                    Image(systemName: depthPipeline.isScanning
                                          ? "stop.fill" : "waveform.circle.fill")
                                    Text(depthPipeline.isScanning ? "Stop" : "Scan")
                                }
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(depthPipeline.isScanning ? .red : .blue)

                            Spacer()

                            Button {
                                if voiceManager.isListening {
                                    voiceManager.stopListening()
                                } else {
                                    voiceManager.startListening()
                                }
                            } label: {
                                Image(systemName: voiceManager.isListening
                                      ? "mic.fill" : "mic.slash")
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

                        // Target selection
                        Text("Target Objects").font(.caption).bold()

                        Button { showingPicker1 = true } label: {
                            HStack {
                                if class1Selection >= 0 {
                                    Circle().fill(colorForClass(class1Selection))
                                        .frame(width: 10, height: 10)
                                    Text(className(for: class1Selection))
                                } else {
                                    Text("Target 1: None").foregroundColor(.secondary)
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

                        Button { showingPicker2 = true } label: {
                            HStack {
                                if class2Selection >= 0 {
                                    Circle().fill(colorForClass(class2Selection))
                                        .frame(width: 10, height: 10)
                                    Text(className(for: class2Selection))
                                } else {
                                    Text("Target 2: None").foregroundColor(.secondary)
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

                        LabeledSlider(title: "Conf", value: $confidence,
                                      range: 0.05...0.95, format: "%.2f")
                            .onChange(of: confidence) { _, v in
                                depthPipeline.detector.confidenceThreshold = Float(v)
                            }

                        LabeledSlider(title: "Sweep", value: $sweepSeconds,
                                      range: 0.5...8.0, format: "%.1fs")

                        Spacer(minLength: 4)

                        Text(String(format: "%.0f FPS", depthPipeline.fps))
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
            depthPipeline.detector.confidenceThreshold = Float(confidence)
            audio.start()
            setupVoiceCallbacks()
            voiceManager.requestPermissions { _ in }
        }
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

    // MARK: - Scan

    private func triggerScan() {
        guard !depthPipeline.isScanning else { return }

        depthPipeline.triggerScan(sweepSeconds: sweepSeconds,
            onColumn: { col, classId, centroid, coverage, z01, pan in
                if classId >= 0 {
                    self.audio.setTarget(classId: classId, centroid: centroid,
                                         coverage: coverage, depth: z01, pan: pan)
                } else {
                    self.audio.clearTarget(pan: pan)
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

    // MARK: - Class management

    private func updateActiveClasses() {
        var ids: [Int] = []
        if class1Selection >= 0 { ids.append(class1Selection) }
        if class2Selection >= 0 && class2Selection != class1Selection {
            ids.append(class2Selection)
        }
        depthPipeline.setActiveClasses(ids)
        if ids.isEmpty { audio.deactivateAll() }
    }

    private func setupVoiceCallbacks() {
        voiceManager.onClassesChanged = { classIds in
            DispatchQueue.main.async {
                self.class1Selection = classIds.count > 0 ? classIds[0] : -1
                self.class2Selection = classIds.count > 1 ? classIds[1] : -1
            }
            self.depthPipeline.setActiveClasses(classIds)
            if classIds.isEmpty { self.audio.deactivateAll() }
        }
        voiceManager.onSweepRateChanged = { multiplier in
            self.sweepSeconds = 2.0 * Double(multiplier)
        }
        voiceManager.onScanTriggered = {
            DispatchQueue.main.async { self.triggerScan() }
        }
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
                            Circle()
                                .fill(Color(hue: Double(cls.id * 37 % 360) / 360.0,
                                            saturation: 0.8, brightness: 0.9))
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
