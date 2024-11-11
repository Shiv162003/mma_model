import csv
input_filename = "output.csv"
output_filename = "output1.csv"
with open(input_filename, "r", newline="", encoding="utf-8") as infile, open(output_filename, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["Classification", "Target Community", "Rationales"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        words = row["labels"].split()
        num_words = len(words)
        new_row = {**row, "Classification": words[0], "Target Community": words[1] if num_words > 1 else "", "Rationales": words[2] if num_words > 2 else ""}
        writer.writerow(new_row)