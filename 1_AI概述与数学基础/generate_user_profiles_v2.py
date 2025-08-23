#!/usr/bin/env python3

import random
import pandas as pd
from datetime import datetime
from pathlib import Path

# --------------------------------------------------
# 1. 配置字段取值范围
# --------------------------------------------------
SEX_CATS         = ["男", "女"]
CITY_CATS        = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安", "南京", "重庆",
                   "苏州", "长沙", "青岛", "天津", "合肥", "郑州", "其他"]
OS_CATS          = ["iOS", "Android", "HarmonyOS", "未透露"]
CONSUMPTION_CATS = ["低", "中低", "中", "中高", "高"]
PAYMENT_CATS     = ["微信", "支付宝", "银行卡", "花呗/白条", "其他"]
INTEREST_POOL    = ["数码", "美妆", "健身", "旅行", "美食", "阅读", "游戏", "音乐", "理财", "摄影"]

# --------------------------------------------------
# 2. 年龄生成：18-40 岁集中，8-80 岁边界
# --------------------------------------------------
def random_age():
    """18-40 岁占 ~70%，8-17 岁和 41-80 岁占 ~30%"""
    if random.random() < 0.7:
        return int(random.normalvariate(30, 8))   # 均值29，标准差6
    else:
        return random.randint(8, 80)

def clip_age(age):
    """确保在 8-80 之间"""
    return max(8, min(80, age))

# --------------------------------------------------
# 3. 单条用户生成函数
# --------------------------------------------------
def generate_user(uid: int):
    age = clip_age(random_age())
    return {
        "user_id"     : f"U{uid:05d}",
        "sex"         : random.choice(SEX_CATS),
        "age"         : age,
        "city"        : random.choice(CITY_CATS),
        "os"          : random.choice(OS_CATS),
        "consumption" : random.choice(CONSUMPTION_CATS),
        "payment"     : random.choice(PAYMENT_CATS),
        "active_days" : min(365, int(random.expovariate(0.01))),
        "balance"     : round(random.lognormvariate(7.0, 1.2), 2),
        "interests"   : "|".join(random.sample(INTEREST_POOL, k=random.randint(1, 3))),
        "created_at"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# --------------------------------------------------
# 4. 批量生成 & 导出
# --------------------------------------------------
def generate_dataset(n_users: int = 500) -> pd.DataFrame:
    data = [generate_user(uid) for uid in range(1, n_users + 1)]
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(500)
    out_path = Path("user_profiles_v2.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已生成 {len(df)} 条用户画像数据，保存至：{out_path.resolve()}")