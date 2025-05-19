import streamlit as st
from google.cloud import bigquery
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import pandas as pd

# --- Initialize the Sentence Transformer model and translator ---

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME, device='cpu')

translator = Translator()
top_k_matches = 30
SIMILARITY_THRESHOLD = 0.6
project_id = "feisty-parity-457910-h8"  # Replace with your GCP project ID
dataset_name = "corpus"  # set your dataset


def search_paragraphs(query, project_id, dataset_name, credentials, top_k=top_k_matches):
    """
    Searches for paragraphs in BigQuery that are semantically similar to the query.

    Args:
        query (str): The search query.
        project_id (str): Your Google Cloud Project ID.
        dataset_name (str): The name of the BigQuery dataset.
        credentials (dict): Google Cloud credentials from Streamlit secrets.
        top_k (int, optional): The number of top matching paragraphs to return.
            Defaults to 30.

    Returns:
        pandas.DataFrame: A DataFrame containing the top matching paragraphs,
        their scores, and other metadata.
    """
    try:
        client = bigquery.Client(project=project_id, credentials=credentials)
    except Exception as e:
        st.error(f"Error initializing BigQuery client: {e}")
        return None

    # Construct the query to retrieve paragraphs and their embeddings.
    query_embedding = model.encode(query)
    sql_query = f"""
        SELECT
            doc_name,
            page_number,
            paragraph_id,
            paragraph,
            language,
            embedding
        FROM
            `{project_id}.{dataset_name}.paragraphs_with_embeddings`
        """
    try:
        query_job = client.query(sql_query)
        results = query_job.result().to_dataframe()
    except Exception as e:
        st.error(f"Error querying BigQuery: {e}")
        return None

    if results.empty:
        st.warning("No paragraphs found in BigQuery.")
        return pd.DataFrame()

    # Convert the embedding from a list to a numpy array
    results["embedding"] = results["embedding"].apply(lambda x: np.array(x))

    para_embeddings = np.vstack(results["embedding"].to_numpy())
    scores = cosine_similarity([query_embedding], para_embeddings)[0]
    results["score"] = scores
    df_filtered = results[results["score"] >= SIMILARITY_THRESHOLD]
    df_top = df_filtered.sort_values(by="score", ascending=False).head(top_k).copy()
    st.write(f"\n🔍 Top {len(df_top)} matches for query: '{query}'\n")
    if df_top.empty:
        st.warning("No matching paragraphs found.")
        return pd.DataFrame()

    return df_top


def filter_best_per_language(df_results, original_query, project_id, dataset_name, credentials):
    """
    Filters the search results to find the best matching paragraph for each language.

    Args:
        df_results (pandas.DataFrame): The DataFrame returned by search_paragraphs.
        original_query (str): The original search query.
        project_id (str): Your Google Cloud Project ID.
        dataset_name (str): The name of the BigQuery dataset.
        credentials (dict): Google Cloud credentials from Streamlit secrets.

    Returns:
        pandas.DataFrame: A DataFrame containing the best matching paragraph for
        each language.
    """
    if df_results.empty:
        st.warning("No results found for the query!")
        return df_results
    client = bigquery.Client(project=project_id, credentials=credentials)
    translated_query = translator.translate(original_query, dest="en").text
    translated_query_embedding = model.encode(translated_query)
    ref_page = df_results.iloc[0]["page_number"]
    best_by_lang = {}

    for _, row in df_results.iterrows():
        para = row["paragraph"]
        lang = row["language"]
        if abs(row["page_number"] - ref_page) > 3:
            continue
        try:
            translated_para = translator.translate(para, dest="en").text
            translated_para_embedding = model.encode(translated_para)
            sim = cosine_similarity(
                [translated_query_embedding], [translated_para_embedding]
            )[0][0]
            if lang not in best_by_lang or sim > best_by_lang[lang]["similarity"]:
                best_by_lang[lang] = {
                    "row": row,
                    "translated": translated_para,
                    "similarity": sim,
                }
        except Exception as e:
            st.error(f"⚠️ Translation failed for lang {lang}: {e}")
            continue
    filtered_rows = [entry["row"] for entry in best_by_lang.values()]
    df_filtered = pd.DataFrame(filtered_rows)
    df_filtered["translated_paragraph"] = df_filtered["paragraph"].apply(
        lambda para: translator.translate(para, dest="en").text
    )
    st.write(f"\n🌍 Filtered top matching paragraph per language:\n")
    return df_filtered


def main():
    st.title("BigQuery Paragraph Search")
    st.write("Enter a query to search for similar paragraphs in BigQuery.")

    # Get credentials from Streamlit secrets
    try:
        gcp_credentials = st.secrets["gcp_credentials"]
    except KeyError:
        st.error("Please configure your Google Cloud credentials in Streamlit secrets.")
        return

    # Search box
    search_query = st.text_area(
        "Enter your query (CZECH EXAMPLE):",
        """Kompresor se obvykle dodává v polyethylenovém nebo jiném
obalu. Pokud použijete k odstranění obalu nůž, dejte pozor,
abyste nepoškodili vnější nátěr kompresoru""",
    )

    if st.button("Search"):
        if not search_query:
            st.warning("Please enter a search query.")
            return

        # Search from BigQuery
        df_results = search_paragraphs(
            search_query, project_id, dataset_name, gcp_credentials
        )
        if df_results is not None and not df_results.empty:
            df_filtered = filter_best_per_language(
                df_results, search_query, project_id, dataset_name, gcp_credentials
            )
            st.dataframe(df_filtered)
        elif df_results is not None:
            st.write("No results to display.")


if __name__ == "__main__":
    main()
