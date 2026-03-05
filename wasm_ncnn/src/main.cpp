#include <emscripten.h>
#include <emscripten/bind.h>
#include <net.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "stb_image.h"
#include "stb_image_write.h"

constexpr int MAX_DETECTIONS = 20; 
constexpr int FLOATS_PER_DETECTION = 5; 
constexpr int  MAX_PLATE_LEN = 8;
constexpr size_t PLATE_BUFFER_SIZE =
    MAX_DETECTIONS * (MAX_PLATE_LEN + 1);
const size_t RESULT_BUFFER_SIZE = MAX_DETECTIONS * FLOATS_PER_DETECTION * sizeof(float);
const float NORM_VALUES[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};

const int TARGET_H = 48;

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

static std::vector<std::string> load_keys(const std::string& path)
{
    std::vector<std::string> keys;
    keys.push_back("blank");  // CTC blank

    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line))
    {
        if (!line.empty())
            keys.push_back(line);
    }
    return keys;
}

static std::string ctc_decode(const ncnn::Mat& out,
                              const std::vector<std::string>& keys)
{
    std::string result;

    int T = out.h;
    int C = out.w;

    const ncnn::Mat& mat = out.channel(0);  // VERY IMPORTANT

    int last_index = 0;

    for (int t = 0; t < T; t++)
    {
        const float* row = mat.row(t);

        int max_index = 0;
        float max_score = row[0];

        for (int i = 1; i < C; i++)
        {
            if (row[i] > max_score)
            {
                max_score = row[i];
                max_index = i;
                std::cout << max_index << " ";
            }
        }

        if (max_index != 0 && max_index != last_index)
        {
            if (max_index < (int)keys.size())
                result += keys[max_index];
        }

        last_index = max_index;
    }

    return result;
}
const int MODEL_W = 320;
const int MODEL_H = 320;
ncnn::Net yolo_net;
ncnn::Net paddle_net;
std::vector<std::string> keys;
extern "C" {
    void models_init() {
        yolo_net.opt.lightmode = true;
        yolo_net.opt.num_threads = 1;
        if (yolo_net.load_param("assets/yolo.param") || yolo_net.load_model("assets/yolo.bin")) {
            return;
        }
        paddle_net.opt.lightmode = true;
        paddle_net.opt.num_threads = 1;
        if (paddle_net.load_param("assets/paddle.param") || paddle_net.load_model("assets/paddle.bin")) {
            return;
        }
        keys = load_keys("assets/dict.txt");
    }

void process_frame(uint8_t* image_data, int orig_w, int orig_h, float conf_thresh, float* resultbuffer, char* platebuffer) {
    memset(resultbuffer, 0, RESULT_BUFFER_SIZE);
    memset(platebuffer, 0, PLATE_BUFFER_SIZE);
    ncnn::Mat input = mat_from_stb(image_data, orig_w, orig_h);
    
    ncnn::Extractor ex = yolo_net.create_extractor();
    ex.set_light_mode(true);
    ex.input("images", input);

    ncnn::Mat out;
    ex.extract("output0", out);
    std::vector<Detection> detections;
    decode_yolov8(out, conf_thresh, detections);
    nms(detections, 0.45f); 
    float sx = orig_w / (float)MODEL_W;
    float sy = orig_h / (float)MODEL_H;
    int count = 0;
    int index = 0;
    for (auto& d : detections) {
        if (count >= MAX_DETECTIONS) break;
        d.bbox.x *= sx;
        d.bbox.y *= sy;
        d.bbox.w *= sx;
        d.bbox.h *= sy;

        // Save cropped images (stb_image_write used for saving)
        int x = (int)d.bbox.x;
        int y = (int)d.bbox.y;
        int w = (int)d.bbox.w;
        int h = (int)d.bbox.h;

        ncnn::Mat cropped = ncnn::Mat::from_pixels_roi(
            image_data,
            ncnn::Mat::PIXEL_RGBA2RGB,
            orig_w,
            orig_h,
            x, y, w, h
        );

        ncnn::Mat resized;
        
        int target_w = std::min(int((float)w * TARGET_H / h), 160);
        ncnn::resize_bilinear(cropped, resized, target_w, TARGET_H);

        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
        resized.substract_mean_normalize(mean_vals, norm_vals);

        // 4️⃣ Feed into OCR
        ncnn::Extractor ocr_ex = paddle_net.create_extractor();
        ocr_ex.input("in0", resized);

        ncnn::Mat ocr_out;
        ocr_ex.extract("out0", ocr_out);
        std::string text = ctc_decode(ocr_out, keys);

        resultbuffer[index++] = d.confidence;
        resultbuffer[index++] = d.bbox.x;
        resultbuffer[index++] = d.bbox.y;
        resultbuffer[index++] = d.bbox.w;
        resultbuffer[index++] = d.bbox.h;

        char* plate_slot = platebuffer + count * (MAX_PLATE_LEN + 1);

        strncpy(plate_slot, text.c_str(), MAX_PLATE_LEN);
        plate_slot[MAX_PLATE_LEN] = '\0';  // guarantee null termination

        count++;
    }
}
}

EMSCRIPTEN_BINDINGS(model) {
    emscripten::function("models_init", & models_init);
}

EMSCRIPTEN_BINDINGS(process) {
    emscripten::function("process_frame", &process_frame, emscripten::allow_raw_pointer<uint8_t>());
}
