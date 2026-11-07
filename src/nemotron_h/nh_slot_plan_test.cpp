#include "strata/nemotron_h/slot_plan.hpp"

#include <cstdio>
#include <vector>

int main() {
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::fprintf(stderr, "FAIL: %s\n", message);
            ++failures;
        }
    };

    const std::vector<double> broad = {10, 10, 10, 10};
    const std::vector<double> skewed = {100, 1, 1, 1};
    const std::vector<strata::nemotron_h::ProfiledLayer> layers = {
        {100, 4, &broad},
        {200, 4, &skewed},
    };
    const std::vector<int> slots = strata::nemotron_h::profile_slot_plan(layers, 300);
    check(slots.size() == 2, "one allocation count per layer");
    check(slots[0] == 1 && slots[1] == 1, "the plan selects the highest profile mass per byte");

    const std::vector<int> empty = strata::nemotron_h::profile_slot_plan(layers, 99);
    check(empty[0] == 0 && empty[1] == 0, "the plan never exceeds the byte budget");

    if (failures == 0) std::puts("nh_slot_plan_test: OK");
    return failures == 0 ? 0 : 1;
}
