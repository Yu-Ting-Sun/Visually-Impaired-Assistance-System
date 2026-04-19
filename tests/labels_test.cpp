#include "Labels.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{

void Expect(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void TestIndoorCommon50Labels()
{
    std::vector<std::string> labels;
    Expect(GetLabelsVector(labels), "GetLabelsVector should succeed");
    Expect(labels.size() == 50, "indoor_common_50 deployment should expose exactly 50 labels");
    Expect(labels[0] == "person", "label[0] should be person");
    Expect(labels[1] == "chair", "label[1] should be chair");
    Expect(labels[2] == "table", "label[2] should be table");
    Expect(labels[4] == "couch", "label[4] should be couch");
    Expect(labels[24] == "backpack", "label[24] should be backpack");
    Expect(labels[28] == "suitcase", "label[28] should be suitcase");
    Expect(labels[49] == "scissors", "label[49] should be scissors");
}

} // namespace

int main()
{
    TestIndoorCommon50Labels();
    std::cout << "labels_test: PASS\n";
    return 0;
}
