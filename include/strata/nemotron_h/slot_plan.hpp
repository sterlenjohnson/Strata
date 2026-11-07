#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace strata::nemotron_h {

struct ProfiledLayer {
    size_t slot_bytes;
    int experts;
    const std::vector<double>* profile;
};

inline std::vector<int> profile_slot_plan(const std::vector<ProfiledLayer>& layers, size_t budget) {
    struct Candidate {
        double score;
        size_t bytes;
        int layer;
    };
    std::vector<Candidate> candidates;
    for (size_t li = 0; li < layers.size(); ++li) {
        const ProfiledLayer& layer = layers[li];
        if (layer.slot_bytes == 0 || layer.experts <= 0 || layer.profile == nullptr) continue;
        std::vector<double> scores = *layer.profile;
        scores.resize((size_t) layer.experts, 0.0);
        std::sort(scores.begin(), scores.end(), std::greater<double>());
        for (int expert = 0; expert < layer.experts; ++expert)
            candidates.push_back({scores[(size_t) expert], layer.slot_bytes, (int) li});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        const long double ar = (long double) a.score / a.bytes;
        const long double br = (long double) b.score / b.bytes;
        if (ar != br) return ar > br;
        if (a.score != b.score) return a.score > b.score;
        return a.layer < b.layer;
    });
    std::vector<int> slots(layers.size(), 0);
    for (const Candidate& candidate : candidates) {
        if (candidate.bytes > budget) continue;
        budget -= candidate.bytes;
        ++slots[(size_t) candidate.layer];
    }
    return slots;
}

} // namespace strata::nemotron_h
