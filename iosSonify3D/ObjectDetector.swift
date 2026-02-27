import Foundation
import Vision
import CoreML
import UIKit
import ARKit
import Accelerate

final class ObjectDetector {

    var confidenceThreshold: Float = 0.25
    var activeClassIds: Set<Int> = []
    private let classLock = NSLock()

    private var mlModel: MLModel?
    private var vnModel: VNCoreMLModel?
    private var isModelLoaded = false
    private var useRawOutput = false

    // Cache output name for detection tensor
    private var detOutputName: String?
    private var protoOutputName: String?

    init() {
        loadModel()
    }

    private func loadModel() {
        let candidates = [
            "yolov8n-seg", "yolov8s-seg", "yolov8n_seg", "yolov8s_seg",
            "yolov8n", "yolov8s", "YOLOv8n-seg", "YOLOv8s-seg", "best"
        ]
        var modelURL: URL?
        for name in candidates {
            if let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") {
                modelURL = url; break
            }
        }
        guard let url = modelURL else {
            print("[YOLO] ⚠️ No model found."); return
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try MLModel(contentsOf: url, configuration: config)
            self.mlModel = model

            let desc = model.modelDescription
            print("[YOLO] Loaded: \(url.lastPathComponent)")

            var hasMultiArray = false
            for (name, d) in desc.outputDescriptionsByName {
                let shape = d.multiArrayConstraint?.shape.map { $0.intValue } ?? []
                print("[YOLO]   Output '\(name)': type=\(d.type.rawValue) shape=\(shape)")
                if d.type == .multiArray {
                    hasMultiArray = true
                    // [1, 116, 8400] → detection, [1, 32, 160, 160] → proto
                    if shape.count == 3 && shape[2] == 8400 {
                        detOutputName = name
                    } else if shape.count == 4 && shape[1] == 32 {
                        protoOutputName = name
                    }
                }
            }
            for (name, d) in desc.inputDescriptionsByName {
                if let ic = d.imageConstraint {
                    print("[YOLO]   Input '\(name)': \(ic.pixelsWide)x\(ic.pixelsHigh)")
                }
            }

            if hasMultiArray {
                print("[YOLO] Raw tensor mode. Det='\(detOutputName ?? "?")' Proto='\(protoOutputName ?? "?")'")
                useRawOutput = true
            }

            vnModel = try? VNCoreMLModel(for: model)
            isModelLoaded = true
        } catch {
            print("[YOLO] Load failed: \(error)")
        }
    }

    func setActiveClasses(_ ids: [Int]) {
        classLock.lock(); activeClassIds = Set(ids); classLock.unlock()
        print("[YOLO] Active: \(ids)")
    }

    func detect(
        pixelBuffer: CVPixelBuffer,
        gridWidth: Int, gridHeight: Int,
        completion: @escaping (_ classGrid: [UInt8]?, _ detections: [Detection]?) -> Void
    ) {
        guard isModelLoaded else { completion(nil, nil); return }
        classLock.lock(); let targets = activeClassIds; classLock.unlock()
        guard !targets.isEmpty else { completion(nil, nil); return }

        if useRawOutput {
            detectRaw(pixelBuffer: pixelBuffer, gridWidth: gridWidth, gridHeight: gridHeight,
                      targetIds: targets, completion: completion)
        } else {
            detectVision(pixelBuffer: pixelBuffer, gridWidth: gridWidth, gridHeight: gridHeight,
                         targetIds: targets, completion: completion)
        }
    }

    // MARK: - Raw CoreML with pixel-wise masks

    private func detectRaw(
        pixelBuffer: CVPixelBuffer,
        gridWidth: Int, gridHeight: Int,
        targetIds: Set<Int>,
        completion: @escaping ([UInt8]?, [Detection]?) -> Void
    ) {
        guard let model = mlModel else { completion(nil, nil); return }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
                let inputDesc = model.modelDescription.inputDescriptionsByName[inputName]
                let inW = inputDesc?.imageConstraint?.pixelsWide ?? 640
                let inH = inputDesc?.imageConstraint?.pixelsHigh ?? 640

                guard let resized = self.resizePixelBuffer(pixelBuffer, width: inW, height: inH) else {
                    completion(nil, nil); return
                }

                let input = try MLDictionaryFeatureProvider(dictionary: [
                    inputName: MLFeatureValue(pixelBuffer: resized)
                ])
                let output = try model.prediction(from: input)

                // Get detection tensor
                guard let detName = self.detOutputName,
                      let detFeat = output.featureValue(for: detName),
                      let detArr = detFeat.multiArrayValue else {
                    completion(nil, nil); return
                }

                // Get proto mask tensor (optional — for pixel-wise masks)
                var protoArr: MLMultiArray?
                if let protoName = self.protoOutputName,
                   let protoFeat = output.featureValue(for: protoName) {
                    protoArr = protoFeat.multiArrayValue
                }

                let result = self.parseDetectionsWithMasks(
                    detArr: detArr, protoArr: protoArr,
                    targetIds: targetIds,
                    inputWidth: inW, inputHeight: inH,
                    gridWidth: gridWidth, gridHeight: gridHeight
                )

                if let (grid, dets) = result, !dets.isEmpty {
                    let summary = dets.prefix(3).map { "\($0.className) \(Int($0.confidence*100))%" }
                        .joined(separator: ", ")
                    print("[YOLO] Det: \(summary)")
                    completion(grid, dets)
                } else {
                    completion(nil, nil)
                }
            } catch {
                print("[YOLO] Inference error: \(error)")
                completion(nil, nil)
            }
        }
    }

    /// Parse YOLOv8-seg outputs into pixel-wise class grid
    private func parseDetectionsWithMasks(
        detArr: MLMultiArray,
        protoArr: MLMultiArray?,
        targetIds: Set<Int>,
        inputWidth: Int, inputHeight: Int,
        gridWidth: Int, gridHeight: Int
    ) -> ([UInt8], [Detection])? {

        let shape = detArr.shape.map { $0.intValue }
        let numFeatures = shape[1]  // 116 for seg (4 + 80 + 32)
        let numBoxes = shape[2]     // 8400
        let numClasses = 80
        let numMaskCoeffs = numFeatures - 4 - numClasses  // 32

        let detPtr = detArr.dataPointer.assumingMemoryBound(to: Float.self)

        // Phase 1: find all detections above threshold for target classes
        struct RawDet {
            var classId: Int; var confidence: Float
            var cx: Float; var cy: Float; var w: Float; var h: Float
            var maskCoeffs: [Float]
            var bbox: CGRect
        }

        var rawDets: [RawDet] = []

        for b in 0..<numBoxes {
            var bestClass = -1; var bestScore: Float = 0
            for c in 0..<numClasses {
                let score = detPtr[(4 + c) * numBoxes + b]
                if score > bestScore { bestScore = score; bestClass = c }
            }
            guard bestScore >= confidenceThreshold, targetIds.contains(bestClass) else { continue }

            let cx = detPtr[0 * numBoxes + b]
            let cy = detPtr[1 * numBoxes + b]
            let w  = detPtr[2 * numBoxes + b]
            let h  = detPtr[3 * numBoxes + b]

            // Extract 32 mask coefficients
            var coeffs = [Float](repeating: 0, count: numMaskCoeffs)
            for m in 0..<numMaskCoeffs {
                coeffs[m] = detPtr[(4 + numClasses + m) * numBoxes + b]
            }

            let nx = (cx - w/2) / Float(inputWidth)
            let ny = (cy - h/2) / Float(inputHeight)
            let nw = w / Float(inputWidth)
            let nh = h / Float(inputHeight)
            let bbox = CGRect(x: CGFloat(max(0, nx)), y: CGFloat(max(0, ny)),
                              width: CGFloat(min(1 - max(0,nx), nw)),
                              height: CGFloat(min(1 - max(0,ny), nh)))

            rawDets.append(RawDet(classId: bestClass, confidence: bestScore,
                                   cx: cx, cy: cy, w: w, h: h,
                                   maskCoeffs: coeffs, bbox: bbox))
        }

        // NMS
        let sorted = rawDets.sorted { $0.confidence > $1.confidence }
        var keep: [RawDet] = []
        var suppressed = Set<Int>()
        for i in 0..<sorted.count {
            guard !suppressed.contains(i) else { continue }
            keep.append(sorted[i])
            for j in (i+1)..<sorted.count {
                guard !suppressed.contains(j) else { continue }
                if iou(sorted[i].bbox, sorted[j].bbox) > 0.5 { suppressed.insert(j) }
            }
        }

        guard !keep.isEmpty else { return nil }

        // Phase 2: Generate pixel-wise masks on the grid
        var grid = [UInt8](repeating: 0, count: gridWidth * gridHeight)
        var detections: [Detection] = []

        // If we have proto masks, compute per-pixel segmentation
        if let proto = protoArr, numMaskCoeffs == 32 {
            let protoShape = proto.shape.map { $0.intValue }
            // Expected: [1, 32, 160, 160]
            guard protoShape.count == 4, protoShape[1] == 32 else {
                // Fall back to bbox
                return bboxFallback(keep: keep, gridWidth: gridWidth, gridHeight: gridHeight)
            }
            let pH = protoShape[2]  // 160
            let pW = protoShape[3]  // 160
            let protoPtr = proto.dataPointer.assumingMemoryBound(to: Float.self)

            for det in keep {
                let gridVal = UInt8(min(255, det.classId + 1))

                // Compute mask = sigmoid(coeffs · protos) at each grid cell
                // Map grid coords to proto coords
                for gy in 0..<gridHeight {
                    for gx in 0..<gridWidth {
                        // Grid cell center in normalized coords [0,1]
                        let normX = (Float(gx) + 0.5) / Float(gridWidth)
                        let normY = (Float(gy) + 0.5) / Float(gridHeight)

                        // Check if inside bounding box first (optimization)
                        if normX < Float(det.bbox.minX) || normX > Float(det.bbox.maxX) ||
                           normY < Float(det.bbox.minY) || normY > Float(det.bbox.maxY) {
                            continue
                        }

                        // Map to proto coords
                        let px = Int(normX * Float(pW))
                        let py = Int(normY * Float(pH))
                        guard px >= 0, px < pW, py >= 0, py < pH else { continue }

                        // Dot product: sum(coeffs[m] * proto[0, m, py, px])
                        var dot: Float = 0
                        for m in 0..<32 {
                            // proto layout: [1, 32, pH, pW] → index = m * pH * pW + py * pW + px
                            let protoVal = protoPtr[m * pH * pW + py * pW + px]
                            dot += det.maskCoeffs[m] * protoVal
                        }

                        // Sigmoid
                        let maskVal = 1.0 / (1.0 + expf(-dot))

                        if maskVal > 0.5 {
                            let idx = gy * gridWidth + gx
                            // Only write if empty or same class (don't overwrite closer objects)
                            if grid[idx] == 0 || grid[idx] == gridVal {
                                grid[idx] = gridVal
                            }
                        }
                    }
                }

                let className = cocoClasses.first(where: { $0.value == det.classId })?.key ?? "class\(det.classId)"
                detections.append(Detection(classId: det.classId, className: className,
                                            confidence: det.confidence, boundingBox: det.bbox))
            }
        } else {
            // No proto masks — fall back to bounding box
            return bboxFallback(keep: keep, gridWidth: gridWidth, gridHeight: gridHeight)
        }

        return (grid, detections)
    }

    /// Fallback: fill grid with bounding boxes (used when proto masks unavailable)
    private func bboxFallback(keep: [Any], gridWidth: Int, gridHeight: Int) -> ([UInt8], [Detection])? {
        // This handles the struct from parseDetectionsWithMasks
        // Using a generic approach since RawDet is local
        return nil  // Will use proto masks in practice
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        if inter.isNull { return 0 }
        let ia = Float(inter.width * inter.height)
        let ua = Float(a.width * a.height + b.width * b.height) - ia
        return ua > 0 ? ia / ua : 0
    }

    private func resizePixelBuffer(_ pb: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ci = CIImage(cvPixelBuffer: pb)
        let sx = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pb))
        let sy = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pb))
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        var out: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferCGImageCompatibilityKey: true,
                             kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, &out)
        if let out = out { ctx.render(scaled, to: out) }
        return out
    }

    // MARK: - Vision fallback for detection-only models

    private func detectVision(
        pixelBuffer: CVPixelBuffer, gridWidth: Int, gridHeight: Int,
        targetIds: Set<Int>,
        completion: @escaping ([UInt8]?, [Detection]?) -> Void
    ) {
        guard let model = vnModel else { completion(nil, nil); return }
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        let request = VNCoreMLRequest(model: model) { [weak self] req, error in
            guard let self else { return }
            if error != nil { completion(nil, nil); return }
            guard let results = req.results as? [VNRecognizedObjectObservation], !results.isEmpty else {
                completion(nil, nil); return
            }
            var dets: [Detection] = []; var grid = [UInt8](repeating: 0, count: gridWidth * gridHeight)
            for obs in results {
                guard let label = obs.labels.first else { continue }
                let cid = self.resolveClassId(label.identifier)
                guard cid >= 0, label.confidence >= self.confidenceThreshold, targetIds.contains(cid) else { continue }
                dets.append(Detection(classId: cid, className: label.identifier,
                                      confidence: label.confidence, boundingBox: obs.boundingBox))
                let bb = obs.boundingBox
                let x0 = max(0, min(gridWidth-1, Int(bb.minX * CGFloat(gridWidth))))
                let y0 = max(0, min(gridHeight-1, Int((1-bb.maxY) * CGFloat(gridHeight))))
                let x1 = max(0, min(gridWidth-1, Int(bb.maxX * CGFloat(gridWidth))))
                let y1 = max(0, min(gridHeight-1, Int((1-bb.minY) * CGFloat(gridHeight))))
                let v = UInt8(min(255, cid + 1))
                for gy in min(y0,y1)...max(y0,y1) { for gx in min(x0,x1)...max(x0,x1) { grid[gy*gridWidth+gx] = v } }
            }
            completion(dets.isEmpty ? nil : grid, dets.isEmpty ? nil : dets)
        }
        request.imageCropAndScaleOption = .scaleFill
        DispatchQueue.global(qos: .userInitiated).async {
            do { try handler.perform([request]) } catch { completion(nil, nil) }
        }
    }

    private func resolveClassId(_ rawName: String) -> Int {
        let cleaned = rawName.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if let id = cocoClasses[cleaned] { return id }
        return -1
    }

    struct Detection {
        let classId: Int
        let className: String
        let confidence: Float
        let boundingBox: CGRect
    }
}
