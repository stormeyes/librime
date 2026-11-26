#include <cmath>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <rime/config.h>
#include <rime/dict/vocabulary.h>
#include <rime/gear/user_history_lm.h>

using namespace rime;

namespace {

DictEntry UserEntry(const string& text, int commits, double weight) {
  DictEntry e;
  e.text = text;
  e.commit_count = commits;
  e.weight = weight;
  return e;
}

DictEntry SysEntry(const string& text) {
  DictEntry e;
  e.text = text;
  e.weight = -6.0;
  return e;
}

HistoryLmParams On() {
  HistoryLmParams p;
  p.unigram_weight = 1.0;
  p.bigram_weight = 1.5;
  p.priority_min_commits = 2;
  p.priority_min_weight = -11.5;
  return p;
}

string TempPath(const char* name) {
  string p = ::testing::TempDir() + name;
  std::remove(p.c_str());
  return p;
}

}  // namespace

TEST(RimeUserHistoryLmTest, DefaultParamsGiveNoBonus) {
  HistoryLmParams off;
  EXPECT_FALSE(off.enabled());
  UserHistoryLm lm;
  lm.Record("", {"这个", "字段"});
  lm.Record("", {"这个", "字段"});
  auto e = UserEntry("字段", 219, -6.5);
  EXPECT_EQ(0.0, HistoryBonus(off, lm.Successors("这个"), e));
}

TEST(RimeUserHistoryLmTest, UnigramBonusScalesWithCommits) {
  auto p = On();
  double b219 = HistoryBonus(p, nullptr, UserEntry("字段", 219, -6.5));
  double b10 = HistoryBonus(p, nullptr, UserEntry("字段", 10, -6.5));
  EXPECT_NEAR(std::log1p(219 / 4.0), b219, 1e-9);
  EXPECT_GT(b219, b10);
  EXPECT_GT(b10, 0.0);
  // 上限
  EXPECT_DOUBLE_EQ(5.0, HistoryBonus(p, nullptr, UserEntry("字段", 100000, -1)));
}

TEST(RimeUserHistoryLmTest, UnigramGatedLikeUserPhrasePriority) {
  auto p = On();
  // 只上屏过一次
  EXPECT_EQ(0.0, HistoryBonus(p, nullptr, UserEntry("蛋糕", 1, -6.0)));
  // 衰减殆尽（扣除拼写惩罚后低于门槛）
  EXPECT_EQ(0.0, HistoryBonus(p, nullptr, UserEntry("番号", 3, -12.0)));
  // 系统词（commit_count=0）
  EXPECT_EQ(0.0, HistoryBonus(p, nullptr, SysEntry("一段")));
  // 单字
  EXPECT_EQ(0.0, HistoryBonus(p, nullptr, UserEntry("的", 5000, -3.0)));
  // 模糊/简拼匹配
  auto fuzzy = UserEntry("字段", 219, -7.2);
  fuzzy.credibility = -0.7;
  EXPECT_EQ(0.0, HistoryBonus(p, nullptr, fuzzy));
}

TEST(RimeUserHistoryLmTest, BigramNeedsTwoOccurrences) {
  auto p = On();
  p.unigram_weight = 0;
  UserHistoryLm lm;
  lm.Record("", {"明天", "早上", "开会"});
  auto e = SysEntry("早上");
  EXPECT_EQ(0.0, HistoryBonus(p, lm.Successors("明天"), e));
  lm.Record("", {"明天", "早上"});
  EXPECT_NEAR(1.5 * std::log1p(2.0), HistoryBonus(p, lm.Successors("明天"), e),
              1e-9);
  EXPECT_EQ(nullptr, lm.Successors("早上开会"));
  EXPECT_EQ(0.0, HistoryBonus(p, lm.Successors("后天"), e));
}

TEST(RimeUserHistoryLmTest, OlderPoolsWeighLess) {
  UserHistoryLm lm;
  lm.Record("", {"明天", "早上"});
  EXPECT_DOUBLE_EQ(1.0, lm.Lookup("明天", "早上").first);
  for (size_t i = 0; i < UserHistoryLm::kPoolSizes[0]; ++i)
    lm.Record("", {"填充", "x"});
  auto r = lm.Lookup("明天", "早上");
  EXPECT_DOUBLE_EQ(0.5, r.first);
  EXPECT_EQ(1, r.second);
  for (size_t i = 0; i < UserHistoryLm::kPoolSizes[1]; ++i)
    lm.Record("", {"填充", "x"});
  EXPECT_DOUBLE_EQ(0.25, lm.Lookup("明天", "早上").first);
  for (size_t i = 0; i < UserHistoryLm::kPoolSizes[2]; ++i)
    lm.Record("", {"填充", "x"});
  EXPECT_EQ(nullptr, lm.Successors("明天"));
  EXPECT_EQ(UserHistoryLm::kCapacity, lm.size());
}

TEST(RimeUserHistoryLmTest, PersistsAndSkipsBadLines) {
  string path = TempPath("history_lm_persist.txt");
  {
    UserHistoryLm lm(path);
    lm.Load();
    lm.Record("zhe ge zi duan", {"这个", "字段"});
    lm.Record("ming tian zao shang", {"明天", "早上"});
  }
  {
    std::ofstream out(path, std::ios::app);
    out << "garbage-without-words\n";
  }
  UserHistoryLm lm(path);
  lm.Load();
  EXPECT_EQ(2u, lm.size());
  EXPECT_EQ(1, lm.Lookup("这个", "字段").second);
  EXPECT_EQ(1, lm.Lookup("明天", "早上").second);
  std::ifstream in(path);
  string line;
  int n = 0;
  while (std::getline(in, line))
    ++n;
  EXPECT_EQ(3, n);  // 坏行留在文件里，加载时跳过
}

TEST(RimeUserHistoryLmTest, LoadCompactsOversizedFile) {
  string path = TempPath("history_lm_compact.txt");
  {
    std::ofstream out(path);
    out << "a b\t旧\t词\n";
    for (size_t i = 0; i < UserHistoryLm::kCapacity * 2; ++i)
      out << "a b\t新\t词\n";
  }
  UserHistoryLm lm(path);
  lm.Load();
  EXPECT_EQ(UserHistoryLm::kCapacity, lm.size());
  EXPECT_EQ(nullptr, lm.Successors("旧"));
  std::ifstream in(path);
  string line;
  size_t n = 0;
  while (std::getline(in, line))
    ++n;
  EXPECT_EQ(UserHistoryLm::kCapacity, n);
}

TEST(RimeUserHistoryLmTest, StartWordOnlyForShortContext) {
  EXPECT_EQ("字段", HistoryStartWord("字段"));
  EXPECT_EQ("", HistoryStartWord("所以其实不需要"));
  EXPECT_EQ("", HistoryStartWord(""));
}

TEST(RimeUserHistoryLmTest, ConfigDefaultsKeepUpstreamBehavior) {
  Config config;
  HistoryLmConfig c = LoadHistoryLmConfig(&config, "translator");
  EXPECT_FALSE(c.params.enabled());
  EXPECT_FALSE(c.record);
  EXPECT_FALSE(c.active());
  EXPECT_EQ("baiwu_history.txt", c.file);
}

TEST(RimeUserHistoryLmTest, ConfigReadsOverrides) {
  Config config;
  config.SetDouble("translator/history_lm/unigram_weight", 1.2);
  config.SetDouble("translator/history_lm/bigram_cap", 3.0);
  config.SetInt("translator/history_lm/bigram_min_count", 3);
  config.SetBool("translator/history_lm/record", true);
  config.SetInt("translator/user_phrase_priority_min_commits", 2);
  config.SetDouble("translator/user_phrase_priority_min_weight", -11.5);
  HistoryLmConfig c = LoadHistoryLmConfig(&config, "translator");
  EXPECT_DOUBLE_EQ(1.2, c.params.unigram_weight);
  EXPECT_DOUBLE_EQ(3.0, c.params.bigram_cap);
  EXPECT_EQ(3, c.params.bigram_min_count);
  EXPECT_EQ(2, c.params.priority_min_commits);
  EXPECT_DOUBLE_EQ(-11.5, c.params.priority_min_weight);
  EXPECT_TRUE(c.record);
  EXPECT_TRUE(c.active());
}
