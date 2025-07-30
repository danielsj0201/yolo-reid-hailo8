#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include "common/tensors.hpp"
#include "common/math.hpp"
#include "repvgg.hpp"
#include "hailo_xtensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

#define OUTPUT_LAYER_NAME "repvgg_a0_person_reid_2048/fc1"
#define TARGET_ID 1
#define SIMILARITY_THRESHOLD 0.75

static std::vector<float> reference_feature_vector;
static bool reference_vector_initialized = false;

static int udp_socket = -1;
static struct sockaddr_in fpga_addr;
static bool socket_initialized = false;

float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0f);
    float norm1 = std::sqrt(std::inner_product(v1.begin(), v1.end(), v1.begin(), 0.0f));
    float norm2 = std::sqrt(std::inner_product(v2.begin(), v2.end(), v2.begin(), 0.0f));
    return dot / (norm1 * norm2 + 1e-6f);
}

void init_udp_socket() {
    if (!socket_initialized) {
        udp_socket = socket(AF_INET, SOCK_DGRAM, 0);
        memset(&fpga_addr, 0, sizeof(fpga_addr));
        fpga_addr.sin_family = AF_INET;
        fpga_addr.sin_port = htons(5005);
        inet_pton(AF_INET, "192.168.100.200", &fpga_addr.sin_addr);
        socket_initialized = true;
    }
}

void send_udp_message(float cx, float cy, float dist) {
    init_udp_socket();
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%.4f,%.4f,%.2f", cx, cy, dist);
    sendto(udp_socket, buffer, strlen(buffer), 0, (struct sockaddr*)&fpga_addr, sizeof(fpga_addr));
    std::cout << "[UDP] Sent message: " << buffer << std::endl;
}

void re_id(HailoROIPtr roi)
{
    if (!roi->has_tensors())
        return;

    roi->remove_objects_typed(HAILO_MATRIX);
    auto tensor = roi->get_tensor(OUTPUT_LAYER_NAME);
    xt::xarray<float> embedding = common::get_xtensor_float(tensor);
    auto normalized_embedding = common::vector_normalization(embedding);
    roi->add_object(hailo_common::create_matrix_ptr(normalized_embedding));
}

void filter(HailoROIPtr roi)
{
    re_id(roi);

    auto matrices = roi->get_objects_typed(HAILO_MATRIX);
    auto detections = roi->get_objects_typed(HAILO_DETECTION);
    auto ids = roi->get_objects_typed(HAILO_UNIQUE_ID);

    if (matrices.empty() || detections.empty() || ids.empty()) return;

    auto mat = std::dynamic_pointer_cast<HailoMatrix>(matrices[0]);
    auto vec = mat->get_data();
    auto det = std::dynamic_pointer_cast<HailoDetection>(detections[0]);
    auto uid = std::dynamic_pointer_cast<HailoUniqueID>(ids[0]);

    int track_id = uid->get_id();
    float similarity = 0.0f;
    bool reassigned = false;


    static float prev_distance = -1.0f;
    static bool prev_initialized = false;

    if (track_id == TARGET_ID && !reference_vector_initialized) {
        reference_feature_vector = vec;
        reference_vector_initialized = true;
        std::cout << "[INIT] Stored reference feature vector for ID=1" << std::endl;
        return;
    }

    if (track_id != TARGET_ID && reference_vector_initialized) {
        similarity = cosine_similarity(reference_feature_vector, vec);
        if (similarity > SIMILARITY_THRESHOLD) {
            roi->remove_objects_typed(HAILO_UNIQUE_ID);
            roi->add_object(std::make_shared<HailoUniqueID>(TARGET_ID));
            track_id = TARGET_ID;
            reassigned = true;
        }
    }

    if (track_id == TARGET_ID) {
        auto bbox = det->get_bbox();
        float xmin = bbox.xmin();
        float ymin = bbox.ymin();
        float xmax = bbox.xmax();
        float ymax = bbox.ymax();
        float width = bbox.width();
        float height = bbox.height();
        float center_x = (xmin + xmax) / 2.0f;
        float center_y = (ymin + ymax) / 2.0f;
        center_x = 1.0f - center_x;
        float ref_width = 0.1200f;
        float ref_height = 0.5986f;
        float ref_distance = 300.0f;
        float standard = ref_width / ref_height;
        float current_ratio = width / height;
        float distance = (current_ratio >= standard)
            ? ref_distance * (ref_width / width)
            : ref_distance * (ref_height / height);


        if (prev_initialized) {
            if (std::abs(distance - prev_distance) > 50.0f) {
                std::cout << "[WARNING] Distance changed too much (" 
                          << prev_distance << " → " << distance 
                          << "), keeping previous." << std::endl;
                distance = prev_distance;
            }
        }
        prev_distance = distance;
        prev_initialized = true;

        std::cout << "[ID] Track ID = " << track_id << std::endl;
        std::cout << "[DETECTION] Label: " << det->get_label()
                  << ", Conf: " << det->get_confidence() << std::endl;
        std::cout << "[CENTER] (" << center_x << ", " << center_y << ")" << std::endl;
        std::cout << "[DISTANCE] " << distance << " cm" << std::endl;

        if (reassigned)
            std::cout << "↪️ Reassigned to ID=1" << std::endl;
        if (similarity > 0.0f)
            std::cout << "[SIMILARITY] " << similarity << std::endl;

        send_udp_message(center_x, center_y, distance);
    }
}
