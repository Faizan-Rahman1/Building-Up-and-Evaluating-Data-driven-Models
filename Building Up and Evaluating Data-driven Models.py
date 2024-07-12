# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
# Make sure to install tensorflow, you don't have to import it, but you need it installed otherwise
# the keras modules will not function correctly.

# NOTE: The CSV files created are for better viewing as the output is too large to display in the console.


# Part 2.1 - Data Preprocessing
# Loading up the given datasets
books_new = pd.read_csv('books_new.csv')
ratings = pd.read_csv('ratings.csv')

# Merging the datasets on the 'bookId' column
book_ratings = books_new.merge(ratings, on='bookId')
# Display the summary statistics of the merged dataset
summary = book_ratings.describe(include='all')
summary.to_csv('summary.csv')
print(summary)
# Check for missing values
print(book_ratings.isnull().sum())
print()

book_ratings.fillna('N/A', inplace=True)
book_ratings.to_csv('book_ratings.csv')


# Part 2.2 - Top 10 Ratings and Confidence Interval
# Calculate the average ratings for each book
average_ratings = book_ratings.groupby('bookId')['rating'].mean()
# Find the top 10 books with the highest average ratings
top_10_ratings = average_ratings.sort_values(ascending=False).head(10)
print(top_10_ratings)
print()
# Calculate the 95% confidence interval for the average ratings by bootstrapping
# to estimate the confidence interval
bootstrap_means = []
# Perform 1000 bootstrap samples of size 100
for i in range(1000):
    bootstrap_sample = np.random.choice(book_ratings['rating'], size=100, replace=True)
    bootstrap_mean = np.mean(bootstrap_sample)
    bootstrap_means.append(bootstrap_mean)

confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
print("95% Confidence Interval for Average Ratings:", confidence_interval)
print()

# Part 2.3 - Average Ratings and Rating Count
# Calculate the average rating and rating count for each book
average_ratings = book_ratings.groupby('bookId')['rating'].agg(['mean', 'count']).reset_index()
average_ratings.columns = ['bookId', 'average_rating', 'rating_count']
average_ratings.to_csv('average_ratings.csv')

# Plotting the average rating vs. rating count
plt.scatter(average_ratings['average_rating'], average_ratings['rating_count'])
plt.xlabel('Average Rating')
plt.ylabel('Rating Count')
plt.title('Average Rating vs. Rating Count')
plt.show()

# Observations:
# From observing the average_ratings dataframe and the plot, we can see there is no relationship between the average
# rating and the rating count. This is due to the rating count a constant (100) for each book.
# In this case, since the rating count is constant, there is no threshold where the average rating can be
# considered insignificant.


# Part 2.4 - Combined Features and Cosine Similarity
# Define the 'liked' column based on the average rating
average_ratings['liked'] = average_ratings['average_rating'].apply(lambda x: 1 if x >= 3.6 else -1)
average_ratings.to_csv('average_ratings.csv')

features = ['Title', 'Author', 'Genre', 'SubGenre', 'Publisher']

# Combine the features into a single column and add to the dataframe
average_ratings['combined_features'] = books_new[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
average_ratings.to_csv('average_ratings.csv')

# Calculate the cosine similarity between the combined features
cv = CountVectorizer()
feature_matrix = cv.fit_transform(average_ratings['combined_features'])
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim)
cosine_sim_df.to_csv('cosine_sim.csv')


# Part 2.5 - Vector Space Method for Book Recommendation
def vec_space_method(book_title):
    # Find the index of the input book
    try:
        book_index = average_ratings[books_new['Title'] == book_title].index.values[0]
    except IndexError:
        print("Book not found in the dataset.")
        return None
    # Find the cosine similarity of the input book with all other books
    similarities = cosine_sim[book_index]
    # Get the indices of the top 10 most similar books
    similar_books_indices = np.argsort(similarities)[::-1][1:11]  # Exclude the book itself
    # Get the titles of the similar books
    similar_books = books_new.iloc[similar_books_indices]['Title'].tolist()
    return similar_books


# Example usage of vec_space_method
book_title = "Fundamentals of Wavelets"  # Example book title
similar_books = vec_space_method(book_title)
# Error checking for invalid book title
if similar_books is None:
    print("Please enter a valid book title.")
    print()
else:
    print(f"Top 10 books similar to '{book_title}' using vector space method:")
    for i, book in enumerate(similar_books, 1):
        print(f"{i}. {book}")
    print()


# Part 2.6 - K-Nearest Neighbors (KNN) for Book Recommendation
# Function to find k nearest neighbors using KNN algorithm
def knn_similarity(book_title, k=10):
    # Fit KNN model
    knn_model = NearestNeighbors(n_neighbors=k + 1, metric='cosine')  # k+1 to exclude the book itself
    knn_model.fit(feature_matrix)

    # Find index of the input book
    book_index = books_new[books_new['Title'] == book_title].index
    if len(book_index) == 0:
        print("Book not found in the dataset.")
        return None
    book_index = book_index[0]

    # Find k nearest neighbors
    distances, indices = knn_model.kneighbors(feature_matrix[book_index], n_neighbors=k + 1)
    similar_books_indices = indices.squeeze()[1:]  # Exclude the book itself

    # Get titles of similar books
    similar_books = books_new.iloc[similar_books_indices]['Title'].tolist()
    return similar_books


# Example usage of knn_similarity function
similar_books = knn_similarity(book_title)
# Error checking for invalid book title
if similar_books is None:
    print("Please enter a valid book title.")
    print()
else:
    print(f"Top 10 books similar to '{book_title}' using KNN:")
    for i, book in enumerate(similar_books, 1):
        print(f"{i}. {book}")
    print()


# Part 2.7 - Evaluating recommender systems (Coverage and Personalization)
# Define the test set
# NOTE: No error checking is done as we assumed the test set will not change.
test_set = {
    'User 1': 'Fundamentals of Wavelets',
    'User 2': 'Orientalism',
    'User 3': 'How to Think Like Sherlock Holmes',
    'User 4': 'Data Scientists at Work'
}

# Generate recommendations for each user using both recommender systems
recommendations_vec_space = {user: vec_space_method(book_title) for user, book_title in test_set.items()}
recommendations_knn = {user: knn_similarity(book_title) for user, book_title in test_set.items()}


# Function to calculate coverage
def calculate_coverage(recommendations, total_items):
    # Find the unique items recommended by the system
    unique_items = set(item for user_recommendations in recommendations.values() for item in user_recommendations)
    # Calculate the coverage as a percentage of unique items recommended divided by the total items recommended
    coverage = len(unique_items) / total_items * 100
    return coverage


# Personalization - calculate average similarity between all pairs of recommendations
def personalization(recommendations):
    similarities = []
    # Calculate similarity between all pairs of recommendations
    for user1, rec1 in recommendations.items():
        for user2, rec2 in recommendations.items():
            if user1 != user2:
                # Jaccard similarity ((A ∩ B) / (A ∪ B))
                similarity = len(set(rec1) & set(rec2)) / float(len(set(rec1) | set(rec2)))
                similarities.append(similarity)
    # Calculate personalization as 1 - (average similarity)
    return (1 - sum(similarities) / len(similarities)) * 100


# Calculate coverage for both recommender systems
total_items = len(average_ratings)
coverage_vec_space = calculate_coverage(recommendations_vec_space, total_items)
coverage_knn = calculate_coverage(recommendations_knn, total_items)
print('Coverage for Vector Space Method: %.2f' % coverage_vec_space + '%')
print('Coverage for KNN Similarity Method: %.2f' % coverage_knn + '%')
print()

# Calculate personalization for both recommender systems
personalization_vec_space = personalization(recommendations_vec_space)
personalization_knn = personalization(recommendations_knn)
print('Personalization for Vector Space Method: %.2f' % personalization_vec_space + '%')
print('Personalization for KNN Similarity Method: %.2f' % personalization_knn + '%')
print()

# Evaluation of Recommender Systems:

# Coverage Evaluation:
# Both the Vector Space Method and KNN Similarity Method show similar coverage percentages.
# This suggests that both methods are recommending items from a similar portion of the item space.
# The coverage metric is important as it indicates how well the recommender system can recommend items from the entire
# dataset. In this case, both methods have a coverage of around 15%, which means that they are able to recommend a
# significant portion of the items in the dataset.

# Personalization Evaluation:
# The personalization metric measures how diverse the recommendations are across different users.
# Interestingly, both methods yield similar personalization percentages as well.
# This could indicate that the recommendations generated by both methods are equally tailored to individual user
# preferences.
# It could also suggest that the dataset itself may not have enough diversity to significantly impact
# the personalization metric.

# Overall, the evaluation metrics could be different for both methods if we had a larger test set with more unique books
# and users. In this case, the results suggest that both methods are performing similarly in terms of coverage and
# personalization.


# Part 2.8 - Building an Artificial Neural Network (ANN) for Book Recommendation
def predict_like(test_data):
    # Preprocessing Data
    X = average_ratings[['average_rating', 'rating_count']].values
    y = average_ratings['liked'].values
    # Updates the y values of -1 to 0 and 1 to 1, so we can fit a binary model
    y_binary = (y + 1) // 2

    # Define the Keras model
    model = Sequential([
        Dense(16, input_shape=(2,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the Keras model
    model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the Keras model on the dataset
    model.fit(X, y_binary, batch_size=10, epochs=250, validation_split=0.2, shuffle=True, verbose=2)

    # Evaluate the Keras model
    _, accuracy = model.evaluate(X, y_binary, verbose=0)
    print()
    print('Accuracy: %.2f' % (accuracy * 100) + '%')

    # Make predictions with the model for each book in the test set for each user
    predictions = {}
    for user, book in test_data.items():
        book_features = average_ratings[books_new['Title'] == book][['average_rating', 'rating_count']].values
        # Predict the likelihood of the user liking the book
        # The [0][0] is used to extract the scalar value from the 2D array output
        prediction = model.predict(book_features)[0][0]
        predictions[user] = prediction

    return predictions


# Call the function with the test set
user_book_predictions = predict_like(test_set)
# Print predictions for each user
for user, prediction in user_book_predictions.items():
    book = test_set[user]
    print(f"{user} would like the book '{book}' with a likelihood of {prediction * 100:.2f}%")

# The predictions given by the ANN model are not very accurate as the model is trained on a small dataset.
# With more data and tuning, the model could provide more accurate predictions. You can also experiment with
# different architectures and hyperparameters to improve the model's performance, such as epochs, batch size, and
# the number of layers and units in the network.
