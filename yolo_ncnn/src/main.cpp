#include <emscripten.h>
#include <emscripten/bind.h>
#include <net.h>
#include <vector>
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"

const int MAX_DETECTIONS = 6 * 20; 
const size_t RESULT_BUFFER_SIZE = MAX_DETECTIONS * sizeof(float);
const float NORM_VALUES[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
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

void decode_yolov8(const ncnn::Mat& out, float conf_thresh, std::vector<Detection>& detections)
{
    detections.clear();

    int num_preds = out.w;   
    int rows = out.h;       

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

ncnn::Mat mat_from_stb(unsigned char* img_data, int img_w, int img_h)
{
    ncnn::Mat in = ncnn::Mat::from_pixels(img_data, ncnn::Mat::PIXEL_RGBA2BGR,img_w, img_h);
    in.substract_mean_normalize(0, NORM_VALUES);
    return in;
}

ncnn::Net net;

extern "C" {
    void yolo_init() {
        net.opt.lightmode = true;
        net.opt.num_threads = 1;
        if (net.load_param("assets/model.param") || net.load_model("assets/model.bin")) {
            return;
        }
    }

void process_frame(uint8_t* image_data, int orig_w, int orig_h, float conf_thresh, float* resultbuffer) {
    memset(resultbuffer, 0, RESULT_BUFFER_SIZE);
    ncnn::Mat input = mat_from_stb(image_data, orig_w, orig_h);
    
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.input("images", input);
    const unsigned char* img_ptr = input.channel(0);  

    ncnn::Mat out;
    ex.extract("output0", out);
    std::vector<Detection> detections;
    decode_yolov8(out, conf_thresh, detections);
    nms(detections, 0.45f); 
    int index = 0;
    for (auto& d : detections) {
        resultbuffer[index++] = d.class_id;
        resultbuffer[index++] = d.confidence;
        resultbuffer[index++] = d.bbox.x;
        resultbuffer[index++] = d.bbox.y;
        resultbuffer[index++] = d.bbox.w;
        resultbuffer[index++] = d.bbox.h;
    }
}
}

EMSCRIPTEN_BINDINGS(model) {
    emscripten::function("yolo_init", & yolo_init);
}

EMSCRIPTEN_BINDINGS(process) {
    emscripten::function("process_frame", &process_frame, emscripten::allow_raw_pointer<uint8_t>());
}
