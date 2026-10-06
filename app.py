import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# LOAD SAVED MODEL FILES

@st.cache_resource
def load_data():
    """
    Load the movie dataframe and cosine-similarity matrix.

    movies.pkl:
        Contains the processed movie dataframe created in the notebook.

    similarity.joblib:
        Contains the cosine similarity matrix created from the movie tags.
    """
    try:
        with open("movies.pkl", "rb") as file:
            movies_df = pickle.load(file)

        similarity = joblib.load("similarity.joblib")

        # IMPORTANT:
        # similarity is positional (row 0, row 1, ...).
        # The notebook used dropna() without resetting the index.
        # Resetting here makes dataframe positions match similarity rows.

        movies_df = movies_df.reset_index(drop=True)

        return movies_df, similarity

    except FileNotFoundError as e:
        st.error(
            f"File not found: {e.filename}. "
            "Make sure movies.pkl and similarity.joblib are in the same folder as app.py."
        )
        st.stop()

    except ModuleNotFoundError as e:
        if "pyarrow" in str(e).lower():
            st.error(
                "Your movies.pkl was saved using a pandas Arrow string type. "
                "Install PyArrow in the same environment where Streamlit runs:\n\n"
                "pip install pyarrow"
            )
        else:
            st.error(f"Missing Python module: {e}")
        st.stop()

    except Exception as e:
        st.error(f"Could not load the recommendation files: {e}")
        st.stop()


movies_df, similarity = load_data()


# VALIDATE LOADED FILES
if "title" not in movies_df.columns:
    st.error(
        "The movies.pkl file does not contain a 'title' column. "
        "The app cannot create the movie selection list."
    )
    st.stop()

if not isinstance(similarity, np.ndarray):
    similarity = np.asarray(similarity)

if similarity.ndim != 2:
    st.error("similarity.joblib does not contain a 2-dimensional similarity matrix.")
    st.stop()

if similarity.shape[0] != similarity.shape[1]:
    st.error("The similarity matrix must have the same number of rows and columns.")
    st.stop()

if len(movies_df) != similarity.shape[0]:
    st.error(
        f"Model/data size mismatch: movies.pkl has {len(movies_df)} movies, "
        f"but similarity.joblib has {similarity.shape[0]} rows."
    )
    st.stop()



# RECOMMENDATION FUNCTION

def recommend(movie_title, number_of_recommendations=5):
    """
    Return the most similar movies for the selected movie.

    The similarity matrix was generated with cosine_similarity()
    in the notebook, so each row corresponds to one movie.
    """

    # Find the row position, not the old dataframe index label.
    matches = movies_df.index[movies_df["title"] == movie_title].tolist()

    if not matches:
        return []

    movie_index = matches[0]

    # Get similarity scores for the selected movie.
    similarity_scores = similarity[movie_index]

    # Sort movie positions by similarity score.
    movie_indices = np.argsort(similarity_scores)[::-1]

    recommendations = []

    for index in movie_indices:
        # Skip the selected movie itself.
        if index == movie_index:
            continue

        title = movies_df.iloc[index]["title"]
        score = float(similarity_scores[index])

        recommendations.append(
            {
                "title": title,
                "score": score
            }
        )

        if len(recommendations) == number_of_recommendations:
            break

    return recommendations


# FRONTEND
st.title("🎬 Movie Recommendation System")
st.write(
    "Select a movie and click **Recommend** to get the 5 most similar movies."
)

st.divider()


# Create clean movie list for the selectbox.
movie_titles = (
    movies_df["title"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .tolist()
)


# Keep the selected movie in session state so the result remains
# visible after Streamlit reruns.
selected_movie = st.selectbox(
    "Select a movie:",
    movie_titles,
    index=0
)


number_of_recommendations = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=5,
    value=5
)


recommend_clicked = st.button(
    "🍿 Recommend Movies",
    type="primary",
    use_container_width=True
)

# SHOW RECOMMENDATIONS

if recommend_clicked:

    recommendations = recommend(
        selected_movie,
        number_of_recommendations
    )

    if not recommendations:
        st.warning("No recommendations were found for this movie.")
    else:
        st.subheader(f"Movies similar to: {selected_movie}")

        for position, recommendation in enumerate(recommendations, start=1):

            title = recommendation["title"]
            score = recommendation["score"]

            col1, col2 = st.columns([5, 1])

            with col1:
                st.write(f"### {position}. {title}")

            with col2:
                st.metric(
                    "Similarity",
                    f"{score * 100:.1f}%"
                )

            st.divider()


# SIDEBAR

with st.sidebar:


    st.markdown("## 👨‍💻 Pragyanam Tiwari")

    st.caption("Developer • Movie Recommendation System")

    st.markdown(
        "[🔗 GitHub](https://github.com/BloodXtream)  \n"
        "[💼 LinkedIn](https://www.linkedin.com/in/pragyanam-tiwari/)  \n"
        "[📧 Email](mailto:pragyanamtiwarii@gmail.com)"
    )

    st.divider()


    st.markdown("### 🎬 About the Project")

    st.caption(
        "A content-based movie recommendation system using "
        "cosine similarity."
    )

    with st.expander("How it works", expanded=False):

        st.write(
            """
            The recommendation system uses a saved
            cosine-similarity matrix generated from movie tags.

            The movie data combines:

            • Movie overview  
            • Genres  
            • Keywords  
            • Top cast  
            • Director
            """
        )

        st.caption(
            "The combined text is stemmed, vectorized using "
            "CountVectorizer, and compared using cosine similarity."
        )

    st.divider()



    st.markdown("### 📊 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Movies",
            f"{len(movies_df):,}"
        )

    with col2:
        st.metric(
            "Matrix",
            f"{similarity.shape[0]:,}"
        )

    st.caption(
        f"Similarity matrix: "
        f"{similarity.shape[0]:,} × {similarity.shape[1]:,}"
    )

    st.divider()

    st.caption("Built with Python • Pandas • Scikit-learn • Streamlit")