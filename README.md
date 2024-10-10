Create a simple content classifier using “Top 15 Anime And K-Drama Like True Beauty 

Webtoon” as inspiration. Develop a basic script that classifies webtoon descriptions into 

categories (e.g., romance, action, fantasy) using Python and scikit-learn.

o You can use a small dataset of 10-15 webtoon descriptions.

o Implement a basic model (e.g., decision tree or logistic regression) to classify the 

categories and show the output.


1. 𝘚𝘦𝘵𝘶𝘱
𝘞𝘦'𝘭𝘭 𝘧𝘪𝘳𝘴𝘵 𝘤𝘰𝘭𝘭𝘦𝘤𝘵 𝘢 𝘴𝘮𝘢𝘭𝘭 𝘥𝘢𝘵𝘢𝘴𝘦𝘵 𝘰𝘧 𝘸𝘦𝘣𝘵𝘰𝘰𝘯 𝘥𝘦𝘴𝘤𝘳𝘪𝘱𝘵𝘪𝘰𝘯𝘴, 𝘭𝘢𝘣𝘦𝘭 𝘵𝘩𝘦𝘮 𝘪𝘯𝘵𝘰 𝘤𝘢𝘵𝘦𝘨𝘰𝘳𝘪𝘦𝘴, 𝘢𝘯𝘥 𝘵𝘩𝘦𝘯 𝘪𝘮𝘱𝘭𝘦𝘮𝘦𝘯𝘵 𝘢 𝘤𝘭𝘢𝘴𝘴𝘪𝘧𝘪𝘦𝘳 𝘶𝘴𝘪𝘯𝘨 𝘦𝘪𝘵𝘩𝘦𝘳 𝘢 𝘋𝘦𝘤𝘪𝘴𝘪𝘰𝘯 𝘛𝘳𝘦𝘦 𝘰𝘳 𝘓𝘰𝘨𝘪𝘴𝘵𝘪𝘤 𝘙𝘦𝘨𝘳𝘦𝘴𝘴𝘪𝘰𝘯 𝘮𝘰𝘥𝘦𝘭.

2. 𝘋𝘢𝘵𝘢𝘴𝘦𝘵
𝘏𝘦𝘳𝘦’𝘴 𝘢𝘯 𝘦𝘹𝘢𝘮𝘱𝘭𝘦 𝘥𝘢𝘵𝘢𝘴𝘦𝘵 𝘸𝘪𝘵𝘩 𝘥𝘦𝘴𝘤𝘳𝘪𝘱𝘵𝘪𝘰𝘯𝘴 𝘰𝘧 10 𝘸𝘦𝘣𝘵𝘰𝘰𝘯𝘴 𝘢𝘯𝘥 𝘵𝘩𝘦𝘪𝘳 𝘤𝘢𝘵𝘦𝘨𝘰𝘳𝘪𝘦𝘴 (𝘳𝘰𝘮𝘢𝘯𝘤𝘦, 𝘢𝘤𝘵𝘪𝘰𝘯, 𝘧𝘢𝘯𝘵𝘢𝘴𝘺, 𝘦𝘵𝘤.).

𝘞𝘦𝘣𝘵𝘰𝘰𝘯 𝘋𝘦𝘴𝘤𝘳𝘪𝘱𝘵𝘪𝘰𝘯	𝘊𝘢𝘵𝘦𝘨𝘰𝘳𝘺
"𝘈 𝘩𝘪𝘨𝘩 𝘴𝘤𝘩𝘰𝘰𝘭 𝘨𝘪𝘳𝘭 𝘧𝘢𝘭𝘭𝘴 𝘪𝘯 𝘭𝘰𝘷𝘦 𝘸𝘪𝘵𝘩 𝘢 𝘱𝘰𝘱𝘶𝘭𝘢𝘳 𝘣𝘰𝘺."	𝘙𝘰𝘮𝘢𝘯𝘤𝘦
"𝘐𝘯 𝘢 𝘸𝘰𝘳𝘭𝘥 𝘰𝘧 𝘮𝘢𝘨𝘪𝘤, 𝘢 𝘺𝘰𝘶𝘯𝘨 𝘣𝘰𝘺 𝘴𝘦𝘵𝘴 𝘰𝘶𝘵 𝘵𝘰 𝘣𝘦𝘤𝘰𝘮𝘦 𝘵𝘩𝘦 𝘣𝘦𝘴𝘵 𝘸𝘪𝘻𝘢𝘳𝘥."	𝘍𝘢𝘯𝘵𝘢𝘴𝘺
"𝘈 𝘨𝘪𝘳𝘭 𝘪𝘴 𝘵𝘰𝘳𝘯 𝘣𝘦𝘵𝘸𝘦𝘦𝘯 𝘵𝘸𝘰 𝘣𝘰𝘺𝘴 𝘸𝘩𝘰 𝘣𝘰𝘵𝘩 𝘤𝘢𝘳𝘦 𝘧𝘰𝘳 𝘩𝘦𝘳 𝘥𝘦𝘦𝘱𝘭𝘺."	𝘙𝘰𝘮𝘢𝘯𝘤𝘦
"𝘈 𝘴𝘶𝘱𝘦𝘳𝘩𝘦𝘳𝘰 𝘧𝘪𝘨𝘩𝘵𝘴 𝘵𝘰 𝘴𝘢𝘷𝘦 𝘵𝘩𝘦 𝘤𝘪𝘵𝘺 𝘧𝘳𝘰𝘮 𝘷𝘪𝘭𝘭𝘢𝘪𝘯𝘴."	𝘈𝘤𝘵𝘪𝘰𝘯
"𝘈 𝘥𝘦𝘵𝘦𝘤𝘵𝘪𝘷𝘦 𝘴𝘰𝘭𝘷𝘦𝘴 𝘮𝘺𝘴𝘵𝘦𝘳𝘪𝘰𝘶𝘴 𝘤𝘳𝘪𝘮𝘦𝘴 𝘪𝘯 𝘢 𝘣𝘪𝘨 𝘤𝘪𝘵𝘺."	𝘔𝘺𝘴𝘵𝘦𝘳𝘺
"𝘈 𝘱𝘳𝘪𝘯𝘤𝘦 𝘮𝘶𝘴𝘵 𝘱𝘳𝘰𝘵𝘦𝘤𝘵 𝘩𝘪𝘴 𝘬𝘪𝘯𝘨𝘥𝘰𝘮 𝘧𝘳𝘰𝘮 𝘥𝘢𝘳𝘬 𝘧𝘰𝘳𝘤𝘦𝘴."	𝘍𝘢𝘯𝘵𝘢𝘴𝘺
"𝘛𝘸𝘰 𝘣𝘦𝘴𝘵 𝘧𝘳𝘪𝘦𝘯𝘥𝘴 𝘧𝘢𝘭𝘭 𝘧𝘰𝘳 𝘵𝘩𝘦 𝘴𝘢𝘮𝘦 𝘱𝘦𝘳𝘴𝘰𝘯, 𝘭𝘦𝘢𝘥𝘪𝘯𝘨 𝘵𝘰 𝘤𝘰𝘮𝘱𝘭𝘪𝘤𝘢𝘵𝘪𝘰𝘯𝘴."	𝘙𝘰𝘮𝘢𝘯𝘤𝘦
"𝘈 𝘨𝘳𝘰𝘶𝘱 𝘰𝘧 𝘴𝘶𝘳𝘷𝘪𝘷𝘰𝘳𝘴 𝘣𝘢𝘵𝘵𝘭𝘦𝘴 𝘻𝘰𝘮𝘣𝘪𝘦𝘴 𝘪𝘯 𝘢 𝘱𝘰𝘴𝘵-𝘢𝘱𝘰𝘤𝘢𝘭𝘺𝘱𝘵𝘪𝘤 𝘸𝘰𝘳𝘭𝘥."	𝘈𝘤𝘵𝘪𝘰𝘯
"𝘈 𝘨𝘪𝘳𝘭 𝘯𝘢𝘷𝘪𝘨𝘢𝘵𝘦𝘴 𝘩𝘪𝘨𝘩 𝘴𝘤𝘩𝘰𝘰𝘭 𝘸𝘩𝘪𝘭𝘦 𝘣𝘢𝘭𝘢𝘯𝘤𝘪𝘯𝘨 𝘭𝘰𝘷𝘦 𝘢𝘯𝘥 𝘧𝘳𝘪𝘦𝘯𝘥𝘴𝘩𝘪𝘱𝘴."	𝘙𝘰𝘮𝘢𝘯𝘤𝘦
"𝘈 𝘸𝘢𝘳𝘳𝘪𝘰𝘳 𝘦𝘮𝘣𝘢𝘳𝘬𝘴 𝘰𝘯 𝘢 𝘥𝘢𝘯𝘨𝘦𝘳𝘰𝘶𝘴 𝘲𝘶𝘦𝘴𝘵 𝘵𝘰 𝘳𝘦𝘵𝘳𝘪𝘦𝘷𝘦 𝘢 𝘮𝘢𝘨𝘪𝘤𝘢𝘭 𝘢𝘳𝘵𝘪𝘧𝘢𝘤𝘵."	𝘍𝘢𝘯𝘵𝘢𝘴𝘺

4. Output
The script will output the classification report, showing the performance metrics like precision, recall, and F1-score for each category.

              𝘱𝘳𝘦𝘤𝘪𝘴𝘪𝘰𝘯    𝘳𝘦𝘤𝘢𝘭𝘭  𝘧1-𝘴𝘤𝘰𝘳𝘦   𝘴𝘶𝘱𝘱𝘰𝘳𝘵

       𝘈𝘤𝘵𝘪𝘰𝘯    |  1.00   |   1.00      1.00          1
      𝘍𝘢𝘯𝘵𝘢𝘴𝘺        1.00      1.00      1.00          1
      𝘙𝘰𝘮𝘢𝘯𝘤𝘦      1.00      1.00      1.00          1

      𝘢𝘤𝘤𝘶𝘳𝘢𝘤𝘺                           1.00          3
     𝘮𝘢𝘤𝘳𝘰 𝘢𝘷𝘨       1.00      1.00      1.00          3
   𝘸𝘦𝘪𝘨𝘩𝘵𝘦𝘥 𝘢𝘷𝘨       1.00      1.00      1.00          3

 Explanation
TF-IDF Vectorizer: Converts the webtoon descriptions into a numeric format that can be used by the classifier.
Decision Tree Classifier: A basic machine learning model that learns decision rules from the training data.
Classification Report: Shows the performance of the model on test data.
