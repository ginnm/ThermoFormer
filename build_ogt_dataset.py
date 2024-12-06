import pandas as pd
from Bio import SeqIO
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    psr = ArgumentParser()
    psr.add_argument(
        "--fasta_file",
        type=str,
        required=True,
        help="Path to the fasta file",
    )
    psr.add_argument(
        "--ogt_file",
        type=str,
        required=True,
        help="Path to the ogt file",
    )
    psr.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output csv file",
    )
    return psr.parse_args()


def annotate(ogt_file, fasta_file, output_file):
    """
    Annotates sequences in a FASTA file with optimal growth temperature (OGT) information.

    Parameters:
    ogt_file (str): Path to the file containing OGT information. The file should be a tab-separated values (TSV) file with a column named "taxid" for taxonomic IDs and a column named "temperature" for OGT values.
    fasta_file (str): Path to the input FASTA file containing sequences to be annotated.
    output_file (str): Path to the output file where annotated sequences will be saved. The output file will be a CSV file with columns "sequence", "ogt", and "taxid".

    Returns:
    None
    """
    ogt_df = pd.read_table(ogt_file, index_col="taxid")
    ogt_df.index = ogt_df.index.astype(str)
    records = SeqIO.parse(fasta_file, "fasta")
    taxids = set(ogt_df.index)
    bar = tqdm(records)
    count = 0
    with open(output_file, "w") as f:
        f.write("sequence,ogt,taxid\n")
        for record in bar:
            has_tax_id = "TaxID" in record.description
            ogt = None
            if has_tax_id:
                descs = record.description.split()
                tax_id = None
                for desc in descs:
                    if "TaxID" in desc:
                        tax_id = desc.split("=")[1]
                        if tax_id in taxids:
                            ogt = ogt_df.loc[tax_id, "temperature"]
            if ogt is not None:
                count += 1
                f.write(f"{str(record.seq)},{ogt},{tax_id}\n")
                bar.set_postfix_str(f"Found {count} sequences with OGT")


if __name__ == "__main__":
    args = parse_args()
    annotate(args.ogt_file, args.fasta_file, args.output_file)
