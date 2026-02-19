#include <emscripten.h>
#include <emscripten/bind.h>
#include <net.h>
#include <vector>
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"

// Struct for bounding box and detection
struct RectF
{
    float x, y, w, h;
};

struct Detection
{
    int class_id;
    float confidence;
    RectF bbox;
};

float iou(const RectF& a, const RectF& b)
{
    float ax0 = a.x;
    float ay0 = a.y;
    float ax1 = a.x + a.w;
    float ay1 = a.y + a.h;

    float bx0 = b.x;
    float by0 = b.y;
    float bx1 = b.x + b.w;
    float by1 = b.y + b.h;

    float inter_x0 = std::max(ax0, bx0);
    float inter_y0 = std::max(ay0, by0);
    float inter_x1 = std::min(ax1, bx1);
    float inter_y1 = std::min(ay1, by1);

    float inter_w = std::max(0.0f, inter_x1 - inter_x0);
    float inter_h = std::max(0.0f, inter_y1 - inter_y0);

    float inter_area = inter_w * inter_h;
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;

    float union_area = area_a + area_b - inter_area;
    if (union_area <= 0.0f)
        return 0.0f;

    return inter_area / union_area;
}

void nms(std::vector<Detection>& dets, float iou_thresh)
{
    if (dets.empty()) return;

    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> keep;
    for (auto& d : dets)
    {
        bool suppressed = false;
        for (auto& k : keep)
        {
            if (iou(d.bbox, k.bbox) > iou_thresh)
            {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(d);
    }

    dets.swap(keep);
}

// decode function adapted to 5x8400 layout
void decode_yolov8(const ncnn::Mat& out, float conf_thresh, std::vector<Detection>& detections)
{
    detections.clear();

    int num_preds = out.w;   // 8400
    int rows = out.h;        // 5

    if (rows != 5) {
        fprintf(stderr, "Expected 5 rows, got %d\n", rows);
        return;
    }

    for (int i = 0; i < num_preds; i++)
    {
        float cx   = out.row(0)[i];
        float cy   = out.row(1)[i];
        float w    = out.row(2)[i];
        float h    = out.row(3)[i];
        float conf = out.row(4)[i];

        if (conf < conf_thresh) continue;

        Detection det;
        det.class_id = 0;
        det.confidence = conf;
        det.bbox.x = cx - w * 0.5f;
        det.bbox.y = cy - h * 0.5f;
        det.bbox.w = w;
        det.bbox.h = h;

        detections.push_back(det);
    }
}

ncnn::Mat mat_from_stb(unsigned char* img_data, int img_w, int img_h, int target_w, int target_h)
{
    // TODO: IF TARGET IS NOT SAME SIZE
    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(img_data, ncnn::Mat::PIXEL_RGBA2BGR,
    //                                         img_w, img_h,
    //                                         target_w, target_h);
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    ncnn::Mat in = ncnn::Mat::from_pixels(img_data, ncnn::Mat::PIXEL_RGBA2BGR,img_w, img_h);
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in.substract_mean_normalize(0, norm_vals);
    // std::cout << "Original image size: " << img_w << "x" << img_h << std::endl;
    // std::cout << "Resized image size: " << target_w << "x" << target_h << std::endl;
    return in;
}

ncnn::Net net;

extern "C" {
    void yolo_init() {
        // Initialize model for inference
        net.opt.lightmode = true;
        net.opt.num_threads = 1;

        // Load model files
        if (net.load_param("assets/model.param") || net.load_model("assets/model.bin")) {
            return;
        }
    }

    // Main processing function that takes in image data (from JS) and returns bounding boxes (detections)
void process_frame(uint8_t* image_data, int orig_w, int orig_h, float conf_thresh, float* resultbuffer, int target_size) {
    memset(resultbuffer, 0, 6 * 20 * sizeof(float));
    // Create ncnn Mat from the image data
    ncnn::Mat input = mat_from_stb(image_data, orig_w, orig_h, target_size, target_size);
    
    // Initialize ncnn extractor and set light mode
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("images", input);
    // std::cout << "Input mat: w=" << input.w << " h= " << input.h << " c=" << input.c << std::endl;
    const unsigned char* img_ptr = input.channel(0);  // Assuming single channel
    // std::cout << "First 10 pixels of image data: ";
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << static_cast<int>(img_ptr[i]) << " ";
    // }
    // std::cout << std::endl;

    // Extract the output from the model
    ncnn::Mat out;
    ex.extract("output0", out);
    // std::cout << "Raw output data (first few values): ";
    // for (int i = 0; i < std::min(10, out.w); ++i) {
    //     std::cout << out.row(4)[i] << " ";  // This prints confidence values
    // }
    // std::cout << std::endl;

    // Decode YOLOv8 results
    std::vector<Detection> detections;
    decode_yolov8(out, conf_thresh, detections);  // Decode detections based on confidence threshold
    // std::cout << "decode_yolov8 completed" << std::endl;  // Confirm YOLOv8 decoding step is finished

    // Apply Non-Maximum Suppression (NMS)
    nms(detections, 0.45f);  // Filter the detections with NMS
    // std::cout << "nms completed" << std::endl;  // Confirm that NMS has been applied

    // Rescale bounding boxes to original image size
    // TODO: IF ORIGINAL AND TARGET ARE NOT THE SAME
    // float sx = orig_w / (float)target_size;
    // float sy = orig_h / (float)target_size;
    // std::cout << "Rescaling bounding boxes to original size: sx=" << sx << ", sy=" << sy << std::endl;

    int index = 0;
    // Store bounding box data in resultbuffer
    for (auto& d : detections) {
        // d.bbox.x *= sx;
        // d.bbox.y *= sy;
        // d.bbox.w *= sx;
        // d.bbox.h *= sy;

        // std::cout << "Detection " << index + 1 << ":" << std::endl;
        // std::cout << "  Class ID: " << d.class_id << std::endl;
        // std::cout << "  Confidence: " << d.confidence << std::endl;
        // std::cout << "  BBox (x, y, w, h): (" << d.bbox.x << ", " << d.bbox.y << ", " << d.bbox.w << ", " << d.bbox.h << ")" << std::endl;


        // Check for buffer overflow and warn if the buffer size is exceeded
        // if (index + 6 > 600) {
        //     std::cout << "Warning: Result buffer overflow. Index: " << index << ", Max size: 600" << std::endl;
        //     break;
        // }

        // Store bounding box data in the resultbuffer
        resultbuffer[index++] = d.class_id;
        resultbuffer[index++] = d.confidence;
        resultbuffer[index++] = d.bbox.x;
        resultbuffer[index++] = d.bbox.y;
        resultbuffer[index++] = d.bbox.w;
        resultbuffer[index++] = d.bbox.h;
    }

    // std::cout << "process_frame completed" << std::endl;  // Final confirmation
}
}



// Expose functions to JS using EMSCRIPTEN bindings
EMSCRIPTEN_BINDINGS(model) {
    emscripten::function("yolo_init", & yolo_init);
}

EMSCRIPTEN_BINDINGS(process) {
    emscripten::function("process_frame", &process_frame, emscripten::allow_raw_pointer<uint8_t>());
}
