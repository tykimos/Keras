---
layout: post
title:  "손실함수 이야기"
author: Taeyoung, Kim
date:   2017-01-27 23:08:00
categories: Keras
comments: true
---
* mean_squared_error / mse
* mean_absolute_error / mae
* mean_absolute_percentage_error / mape
* mean_squared_logarithmic_error / msle
* squared_hinge
* hinge
* binary_crossentropy: Also known as logloss.
* categorical_crossentropy: Also known as multiclass logloss. Note: using this objective requires that your labels are binary arrays of shape (nb_samples, nb_classes).
* sparse_categorical_crossentropy: As above but accepts sparse labels. Note: this objective still requires that your labels have the same number of dimensions as your outputs; you may need to add a length-1 dimension to the shape of your labels, e.g with np.expand_dims(y, -1).
* kullback_leibler_divergence / kld: Information gain from a predicted probability distribution Q to a true probability distribution P. Gives a measure of difference between both distributions.
* poisson: Mean of (predictions - targets * log(predictions))
* cosine_proximity: The opposite (negative) of the mean cosine proximity between predictions and targets.
