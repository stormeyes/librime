#include <cmath>
#include <gtest/gtest.h>
#include <rime/gear/user_phrase_priority.h>

using rime::UserPhraseKeepsPriority;

TEST(RimeUserPhrasePriorityTest, DisabledByDefaultKeepsUpstreamBehavior) {
  EXPECT_TRUE(UserPhraseKeepsPriority(1, -30.0, 0, -INFINITY));
}

TEST(RimeUserPhrasePriorityTest, SingleCommitLosesAbsolutePriority) {
  EXPECT_FALSE(UserPhraseKeepsPriority(1, -6.0, 2, -11.0));
  EXPECT_TRUE(UserPhraseKeepsPriority(2, -6.0, 2, -11.0));
}

TEST(RimeUserPhrasePriorityTest, DecayedPhraseLosesAbsolutePriority) {
  EXPECT_FALSE(UserPhraseKeepsPriority(3, -11.5, 2, -11.0));
  EXPECT_TRUE(UserPhraseKeepsPriority(219, -6.5, 2, -11.0));
}

