with open("dev.csv", encoding="utf8", errors="ignore") as f1, open("dev_fix.csv", 'w') as f2:
    for line in f1:
        f2.write(line)
