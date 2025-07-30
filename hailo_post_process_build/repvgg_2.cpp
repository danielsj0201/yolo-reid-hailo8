/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
 #include <cmath>  // for std::inner_product
#include <vector>
#include <iostream>
#include "common/tensors.hpp"
#include "common/math.hpp"
#include "repvgg.hpp"
#include "hailo_xtensor.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"

#define OUTPUT_LAYER_NAME "repvgg_a0_person_reid_2048/fc1"

#define OUTPUT_LAYER_NAME "repvgg_a0_person_reid_2048/fc1"
#define TARGET_ID 1
#define SIMILARITY_THRESHOLD 0.85
static std::vector<float> reference_feature_vector;
static bool reference_vector_initialized = false;

//cosine similarity
float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0f);
    float norm1 = std::sqrt(std::inner_product(v1.begin(), v1.end(), v1.begin(), 0.0f));
    float norm2 = std::sqrt(std::inner_product(v2.begin(), v2.end(), v2.begin(), 0.0f));
    return dot / (norm1 * norm2 + 1e-6f); 
}

void re_id(HailoROIPtr roi)
{

    if (!roi->has_tensors())
    {
        return;
    }

    // Remove previous matrices
    roi->remove_objects_typed(HAILO_MATRIX);

    // Convert the tensor to xarray.
    auto tensor = roi->get_tensor(OUTPUT_LAYER_NAME);
    xt::xarray<float> embedding = common::get_xtensor_float(tensor);

    // vector normalization
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

    if (track_id == TARGET_ID && !reference_vector_initialized) {
        reference_feature_vector = vec;
        reference_vector_initialized = true;
        std::cout << "[INIT] Stored reference feature vector for ID=1" << std::endl;
        return;
    }

    if (track_id != TARGET_ID && reference_vector_initialized) {
        float sim = cosine_similarity(reference_feature_vector, vec);
        std::cout << "[COMPARE] ID=" << track_id << " similarity=" << sim << std::endl;

        if (sim > SIMILARITY_THRESHOLD) {
            std::cout << "↪️ Reassigning ID " << track_id << " → " << TARGET_ID << std::endl;

            
            roi->remove_objects_typed(HAILO_UNIQUE_ID);
            roi->add_object(std::make_shared<HailoUniqueID>(TARGET_ID));
        }
    }

    std::cout << "[ID] Track ID = " << track_id << std::endl;
    std::cout << "[DETECTION] Label: " << det->get_label()
              << ", Conf: " << det->get_confidence() << std::endl;
    std::cout << "[FEATURE] ";
    for (size_t i = 0; i < std::min<size_t>(5, vec.size()); ++i)
        std::cout << vec[i] << (i < 4 ? ", " : "...\n");
}
