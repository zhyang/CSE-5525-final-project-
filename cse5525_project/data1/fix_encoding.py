with open("data/SubtaskB_Trial_Test_Labeled.csv", encoding="utf8", errors="ignore") as f1, open("data/dev_b.csv", 'w') as f2:
    for line in f1:
        f2.write(line)
