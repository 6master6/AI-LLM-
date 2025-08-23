// 仅依赖标准库，解析简单 CSV（UTF-8 无引号转义即可）
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct User {
    std::string consumption;   
};

// 把 5 档合并为 3 档
std::string to3(const std::string& c) {
    return (c == "中低" || c == "中高") ? "中" : c;
}

// 统计分布
std::map<std::string, double> distribution(const std::vector<User>& v) {
    std::map<std::string, int> cnt;
    for (const auto& u : v) cnt[to3(u.consumption)]++;
    std::map<std::string, double> p;
    for (const auto& kv : cnt)
        p[kv.first] = static_cast<double>(kv.second) / v.size();
    return p;
}

void print_dist(const std::map<std::string, double>& p, const std::string& title) {
    std::cout << title << ":\n";
    for (const auto& kv : p)
        std::cout << "  " << kv.first << "  " << std::fixed << std::setprecision(3)
                  << kv.second << '\n';
    std::cout << '\n';
}

// 分层抽样
void stratified_split(const std::vector<User>& all,
                      double test_ratio,
                      std::vector<int>& train_idx,
                      std::vector<int>& test_idx,
                      unsigned seed = 42) {
    std::map<std::string, std::vector<int>> buckets;
    for (size_t i = 0; i < all.size(); ++i)
        buckets[to3(all[i].consumption)].push_back(static_cast<int>(i));

    std::mt19937 rng(seed);
    train_idx.clear(); test_idx.clear();
    for (auto& kv : buckets) {
        std::shuffle(kv.second.begin(), kv.second.end(), rng);
        int n_total = kv.second.size();
        int n_test  = std::max<int>(1, n_total * test_ratio);
        for (int i = 0; i < n_total; ++i) {
            if (i < n_test)
                test_idx.push_back(kv.second[i]);
            else
                train_idx.push_back(kv.second[i]);
        }
    }
    std::shuffle(train_idx.begin(), train_idx.end(), rng);
    std::shuffle(test_idx.begin(),  test_idx.end(),  rng);
}

int main() {
    const std::string file = "../../user_profiles_v2.csv";
    std::ifstream fin(file);
    if (!fin) {
        std::cerr << "Cannot open " << file << '\n';
        return 1;
    }

    std::vector<User> all;
    std::string line;
    std::getline(fin, line); // 跳过表头
    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> fields;
        while (std::getline(ss, cell, ',')) fields.push_back(cell);
        if (fields.size() >= 6) {           // 第 6 列是 consumption
            User u{fields[5]};
            all.push_back(std::move(u));
        }
    }

    if (all.empty()) {
        std::cerr << "No data!\n";
        return 1;
    }

    // 黄金标准
    auto gold = distribution(all);
    print_dist(gold, "黄金标准（全量 " + std::to_string(all.size()) + " 条）");

    // 分层 80/20
    std::vector<int> train_idx, test_idx;
    stratified_split(all, 0.2, train_idx, test_idx);

    std::vector<User> train, test;
    for (int i : train_idx) train.push_back(all[i]);
    for (int i : test_idx)  test.push_back(all[i]);

    auto train_dist = distribution(train);
    auto test_dist  = distribution(test);

    print_dist(train_dist, "训练集分布（" + std::to_string(train.size()) + " 条）");
    print_dist(test_dist,  "测试集分布（" + std::to_string(test.size())  + " 条）");

    std::cout << "训练集 | 与黄金标准绝对差:\n";
    for (const auto& kv : gold)
        std::cout << "  " << kv.first << "  "
                  << std::fixed << std::setprecision(3)
                  << std::abs(train_dist[kv.first] - kv.second) << '\n';
    std::cout << '\n';

    std::cout << "测试集 | 与黄金标准绝对差:\n";
    for (const auto& kv : gold)
        std::cout << "  " << kv.first << "  "
                  << std::fixed << std::setprecision(3)
                  << std::abs(test_dist[kv.first] - kv.second) << '\n';

    return 0;
}