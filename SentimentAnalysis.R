## Sentiment analysis of Amazon, Yelp and IMDB


# Read in the data

## test sets from UCI Machine Learning (yelp, amazon and imdb)
## 0 is Negative and 1 is positive

setwd("~/R_projects")

yelp = read.csv("NLP/UCISentiment/yelp_labelled.txt", stringsAsFactors=FALSE, sep="\t", header=FALSE)
amazon = read.csv("NLP/UCISentiment/amazon_cells_labelled.txt", stringsAsFactors=FALSE, sep="\t", header=FALSE)
imdb = read.csv("NLP/UCISentiment/imdb_labelled.txt", stringsAsFactors=FALSE, sep="\t", header=FALSE)

sentiment <- rbind(amazon, yelp, imdb)
sentiment <- sentiment[complete.cases(sentiment),]
str(sentiment)


# Install new packages

install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)


# Create corpus
 
corpus2 = Corpus(VectorSource(sentiment$V1))


# Look at corpus

corpus2[[1]]$content


# Convert to lower-case

corpus2 = tm_map(corpus2, tolower)

corpus2[[1]]

corpus2 = tm_map(corpus2, PlainTextDocument)

# Remove punctuation

corpus2 = tm_map(corpus2, removePunctuation)

corpus2[[1]]$content


# Remove stopwords and apple

corpus2 = tm_map(corpus2, removeWords, c(stopwords("english")))

corpus2[[1]]$content

# Stem document 

corpus2 = tm_map(corpus2, stemDocument)

corpus2[[1]]$content

# Create matrix

frequencies2 = DocumentTermMatrix(corpus2)

frequencies2

# Look at matrix 

inspect(frequencies2[1000:1005,505:515])

# Check for sparsity

findFreqTerms(frequencies2, lowfreq=20)

# Remove sparse terms

# when we remove Sparse terms basically we delete words in a recommendation 
# that are not repeated certain number of times in the corpus. 
# words with negative / positive meaning are penalized (deleted) if not repeated in other comments
# (could reduce sentences by half or more)
# once a word is cut, then it wont play a role in classifiying for sentiment

sparse2 = removeSparseTerms(frequencies2, 0.998)
sparse2

# Convert to a data frame

tweetsSparse2 = as.data.frame(as.matrix(sparse2))

# Make all variable names R-friendly

colnames(tweetsSparse2) = make.names(colnames(tweetsSparse2))

# Add dependent variable

tweetsSparse2$categ = factor(sentiment$V2)

table(tweetsSparse2$categ)

# Split the data

tweetsSparse22 <- tweetsSparse2 #just to be able to find recom later
tweetsSparse22$tweet <- sentiment$V1


install.packages("caTools")
library(caTools)

set.seed(123)

split2 = sample.split(tweetsSparse2$categ, SplitRatio = 0.7)
trainSparse2 = subset(tweetsSparse2, split2==TRUE)
testSparse2 = subset(tweetsSparse2, split2==FALSE)

trainSparse22 = subset(tweetsSparse22, split2==TRUE) #just to be able to find recom later
testSparse22 = subset(tweetsSparse22, split2==FALSE)

table(trainSparse2$categ)
table(testSparse2$categ)

# trainSparse = tweetsSparse

#### Build a CART model  ###########
install.packages("rpart")
install.packages("rpart.port")
library(rpart)
library(rpart.plot)

tweetCART2 = rpart(categ ~ ., data=trainSparse2, method="class")

prp(tweetCART2)


# Evaluate the performance of the model

predictCART2 = predict(tweetCART2, newdata=testSparse2, type="class")

table(testSparse2$categ, predictCART2)

table(testSparse22$categ, predictCART2)

# Compute accuracy

(267+72)/(267+72+12+223)

# Baseline accuracy 

table(testSparse2$categ)

279/(279+295)

###### Random forest model   #############

install.packages("randomForest")
library(randomForest)
set.seed(123)

tweetRF2 = randomForest(categ ~ ., data=trainSparse2)

head(getTree(tweetRF2, 1, labelVar=TRUE))

varImpPlot(tweetRF2) # plot of RF variables by importance

# Make predictions:

predictRF2 = predict(tweetRF2, newdata=testSparse2)

table(testSparse2$categ, predictRF2)


# Accuracy for testSparse2
(221+216)/(221+216+58+79)

#find errors from confussion matrix for Random Forest

testSparse22$tweet[(testSparse22$categ==0)& (predictRF2==1)]
testSparse22$tweet[(testSparse22$categ==1)& (predictRF2==0)]

tweetsSparse2[4,(tweetsSparse2[4,]==1)]
tweetsSparse22$tweet[(tweetsSparse22[4,]==1)]

#find errors from confussion matrix for CART

testSparse22$tweet[(testSparse22$categ=="FALSE")& (predictCART2=="TRUE")]

testSparse22$tweet[(testSparse22$categ=="TRUE")& (predictCART2=="FALSE")]


##### Support Vector Machine Model  #########

install.packages("e1071")
library(e1071)


SVM1N <- svm(categ ~ ., type='C', data=trainSparse2, kernel='radial')

PT1.test2 <- predict(SVM1N, newdata=testSparse2, type="class")
table(testSparse2$categ, PT1.test2)

(171+216)/(171+216+108+79)


grid2 <- tune.svm(categ ~ ., data=trainSparse2, gamma=10^seq(-2,0,by=1), cost=10^seq(0,2,by=1))
summary(grid2)

best.gamma2 <- grid2$best.parameters[[1]]
best.cost2 <- grid2$best.parameters[[2]]

SVM4N <- svm(categ ~ ., type='C', data=trainSparse2, kernel='radial', cost=best.cost2, gamma=best.gamma2)
PT4.test2 <- predict(SVM4N, testSparse2, type="class")
table(testSparse2$categ, PT4.test2)

(114+245)/(114+245+165+50)







