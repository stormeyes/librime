#include <cmath>
#include <gtest/gtest.h>
#include <rime/algo/syllabifier.h>
#include <rime/dict/dictionary.h>
#include <rime/dict/table.h>

using namespace rime;

// 4 字以上词条超出表索引的音节走 match_extra_code；这些音节的拼写惩罚
// （模糊音/简拼）以前被丢掉，「不好意思啊」里第 4 个音节走模糊音也不扣分。
TEST(RimeDictionaryExtraCodeTest, AccumulatesSpellingCredibility) {
  EdgeProperties exact, fuzzy;
  exact.end_pos = 2;
  fuzzy.end_pos = 5;
  fuzzy.type = kFuzzySpelling;
  fuzzy.credibility = std::log(0.01);
  SyllableGraph graph;
  graph.input_length = graph.interpreted_length = 5;
  graph.indices[0][7].push_back(&exact);
  graph.indices[2][8].push_back(&fuzzy);
  SyllableId ids[] = {7, 8};
  table::Code code;
  code.size = 2;
  code.at = ids;
  auto m = dictionary::match_extra_code(&code, 0, graph, 0, false);
  ASSERT_TRUE(m.success);
  EXPECT_EQ(5u, m.end_pos);
  EXPECT_NEAR(std::log(0.01), m.credibility, 1e-9);
}
