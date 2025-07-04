"""
Gemini Ticket Summarizer — OOP Version
---------------------------------------
Reads a ServiceNow ticket DataFrame,
generates an overall summary for each row,
using Gemini LLM in efficient batches,
and returns the final DataFrame with an added summary column.
"""

import pandas as pd
import google.generativeai as genai
import time


class GeminiTicketSummarizer:
    """
    A ServiceNow ticket summarizer powered by Gemini.
    """

    def __init__(self,
                 api_key: str,
                 model_name: str = 'gemini-1.5-flash',
                 batch_size: int = 5,
                 summary_column: str = 'Summary',
                 summary_separator: str = '---',
                 polite_pause: float = 1.0):
        """
        Initialize the summarizer.

        Args:
            api_key (str): Gemini API key.
            model_name (str): Gemini model version.
            batch_size (int): Number of tickets per batch.
            summary_column (str): Name of the output summary column.
            summary_separator (str): Separator to split multiple summaries.
            polite_pause (float): Delay between batches (seconds).
        """
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.summary_column = summary_column
        self.summary_separator = summary_separator
        self.polite_pause = polite_pause

        # Setup Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate_batch_prompt(self, batch_df: pd.DataFrame) -> str:
        """
        Build a single batch prompt for Gemini.

        Args:
            batch_df (pd.DataFrame): Batch of rows.

        Returns:
            str: The complete prompt text.
        """
        prompt = (
            "For each of the following ServiceNow tickets, write an overall summary only, "
            "under 100 words. Combine the ticket details, description, comments, work notes, "
            "and resolution notes into one clear, meaningful summary. Include any important keywords. "
            f"Return each summary in the same order, separated by '{self.summary_separator}'.\n\n"
        )

        for _, row in batch_df.iterrows():
            ticket_text = (
                f"Ticket Number: {row.get('Number', '')}\n"
                f"Account: {row.get('Account', '')}\n"
                f"Short Description: {row.get('Short Description', '')}\n"
                f"Category: {row.get('Category', '')}\n"
                f"Priority: {row.get('Priority', '')}\n"
                f"Description: {row.get('Description', '')}\n"
                f"Additional comments: {row.get('Additional comments', '')}\n"
                f"Work notes: {row.get('Work notes', '')}\n"
                f"Resolution Notes: {row.get('Resolution Notes', '')}\n"
            )
            prompt += f"TICKET:\n{ticket_text}\n"

        prompt += "\nOutput summaries:\n"
        return prompt

    def summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the entire DataFrame in batches and add summaries.

        Args:
            df (pd.DataFrame): Original ticket DataFrame.

        Returns:
            pd.DataFrame: DataFrame with added summaries.
        """
        # Ensure the summary column exists
        if self.summary_column not in df.columns:
            df[self.summary_column] = ""

        total_batches = len(df) // self.batch_size + 1

        for batch_start in range(0, len(df), self.batch_size):
            batch_df = df.iloc[batch_start:batch_start + self.batch_size]

            prompt = self.generate_batch_prompt(batch_df)
            print(f"⚡ Processing batch {batch_start // self.batch_size + 1} of {total_batches}...")

            try:
                response = self.model.generate_content(prompt)
                summaries = response.text.split(self.summary_separator)

                for idx, summary in zip(batch_df.index, summaries):
                    clean_summary = summary.strip()
                    if clean_summary:
                        df.at[idx, self.summary_column] = clean_summary

            except Exception as e:
                print(f"❌ Error in batch {batch_start // self.batch_size + 1}: {e}")

            time.sleep(self.polite_pause)

        return df
# Replace KeyBERT cluster names with Gemini cluster names

def generate_cluster_name_gemini(ticket_texts, genai_model, separator="---"):
    """
    Uses Gemini to generate a short cluster name.
    """
    prompt = (
        "The following are summaries of ServiceNow tickets that belong to the same cluster:\n\n"
    )

    for idx, text in enumerate(ticket_texts, start=1):
        prompt += f"Summary {idx}: {text}\n"

    prompt += (
        "\nBased on the above, provide a short, clear name (max 8-10 words) that describes the main theme of this cluster."
    )

    try:
        response = genai_model.generate_content(prompt)
        cluster_name = response.text.strip().replace("\n", " ")
        return cluster_name
    except Exception as e:
        print(f"❌ Error generating cluster name: {e}")
        return "Uncategorized"


if __name__ == "__main__":
    """
    Example usage:
    1️⃣ Reads your input file
    2️⃣ Runs summarizer
    3️⃣ Saves to a new file
    """

    # ✅ Configs
    INPUT_FILE = "batched_raw_customer_tickets.xlsx"
    OUTPUT_FILE = "batched_raw_customer_tickets_with_overall_summary.xlsx"
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Replace!

    # ✅ Load input data
    df_input = pd.read_excel(INPUT_FILE)

    # ✅ Run summarizer
    summarizer = GeminiTicketSummarizer(
        api_key=GEMINI_API_KEY,
        batch_size=5,
        summary_column='Overall_Summary',
        summary_separator='---',
        polite_pause=1.0
    )

    final_df = summarizer.summarize(df_input)

    # ✅ Save to new Excel
    final_df.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Saved: {OUTPUT_FILE}")
