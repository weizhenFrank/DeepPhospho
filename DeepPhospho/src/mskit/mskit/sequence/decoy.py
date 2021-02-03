protein_list = rk.read_one_col_file()
reverse_protein_list = [_ + '_DecoyReversed' for _ in protein_list]
print(len(protein_list))


total_protein_fasta_path = join_path()
fasta_parser = mskit.seq_process.FastaParser(total_protein_fasta_path)

protein_in_total_fasta = []
for seq_title, seq in fasta_parser.one_protein_generator():
    seq_ident = rk.fasta_title(seq_title)
    if seq_ident in protein_list:
        protein_in_total_fasta.append(seq_title)

len(protein_in_total_fasta)

with open(os.path.splitext(total_protein_fasta_path)[0] + '-DecoyReversed.fasta', 'w') as f:
    for seq_title, seq in fasta_parser.one_protein_generator():
        seq_ident = rk.fasta_title(seq_title)

        if seq_ident in protein_list:
            f.write(seq_title.replace(seq_ident, seq_ident + '_DecoyReversed') + '\n')
            f.write(seq[::-1] + '\n')

        f.write(seq_title + '\n')
        f.write(seq + '\n')
