# Assignment 6 Part 1 - Writeup

**Name:** Max Carmona  
**Date:** 11-20-25

---

## Part 1: Understanding Your Model

### Question 1: R² Score Interpretation
What does the R² score tell you about your model? What does it mean if R² is close to 1? What if it's close to 0?

**YOUR ANSWER: The R² score tells me how well the model matches the actual data. If the R² score is close to 1, the model fits the data well. If the R² score is close to 0, the model does not explain the data well. It basically shows how much of the pattern the model is capturing. **




---

### Question 2: Mean Squared Error (MSE)
What does the MSE (Mean Squared Error) mean in plain English? Why do you think we square the errors instead of just taking the average of the errors?

**YOUR ANSWER: MSE shows how far the model’s predictions are from the true values on average. A lower MSE means the predictions are closer to the real scores. We square the errors so negative and positive mistakes do not cancel each other out, and so large mistakes count more. **




---

### Question 3: Model Reliability
Would you trust this model to predict a score for a student who studied 10 hours? Why or why not? Consider:
- What's the maximum hours in your dataset?
- What happens when you make predictions outside the range of your training data?

**YOUR ANSWER: I would be careful trusting a prediction at 10 hours if the dataset does not include any values near 10. If 10 hours is outside the range of the original data, the model is guessing beyond what it learned. Predictions are most reliable inside the range of the training data. **




---

## Part 2: Data Analysis

### Question 4: Relationship Description
Looking at your scatter plot, describe the relationship between hours studied and test scores. Is it:
- Strong or weak?
- Linear or non-linear?
- Positive or negative?

**YOUR ANSWER: The scatter plot shows a positive relationship: as hours studied increase, test scores also increase. The pattern looks linear and strong because the points follow an upward trend. **




---

### Question 5: Real-World Limitations
What are some real-world factors that could affect test scores that this model doesn't account for? List at least 3 factors.

**YOUR ANSWER:**
1. The student’s understanding of the material
2. Sleep, stress, or health
3. The quality of how they studied
These factors can affect scores but are not included in the data.


---

## Part 3: Code Reflection

### Question 6: Train/Test Split
Why do we split our data into training and testing sets? What would happen if we trained and tested on the same data?

**YOUR ANSWER: We split the data so the model can be tested on new information it has not seen before. If we used the same data for training and testing, the model could appear perfect even though it did not generalize well. **




---

### Question 7: Most Challenging Part
What was the most challenging part of this assignment for you? How did you overcome it (or what help do you still need)?

**YOUR ANSWER: The most challenging part was understanding how each function works together to build the model. Following the ice cream example helped me figure out the steps. **




---

## Part 4: Extending Your Learning

### Question 8: Future Applications
Describe one real-world problem you could solve with linear regression. What would be your:
- **Feature (X): Square footage ** 
- **Target (Y): Price ** 
- **Why this relationship might be linear: Larger houses tend to have higher prices, which forms a pattern that is close to a straight line. **



---

## Grading Checklist (for your reference)

Before submitting, make sure you have:
- [ ] Completed all functions in `a6_part1.py`
- [ ] Generated and saved `scatter_plot.png`
- [ ] Generated and saved `predictions_plot.png`
- [ ] Answered all questions in this writeup with thoughtful responses
- [ ] Pushed all files to GitHub (code, plots, and this writeup)

---

## Optional: Extra Credit (+2 points)

If you want to challenge yourself, modify your code to:
1. Try different train/test split ratios (60/40, 70/30, 90/10)
2. Record the R² score for each split
3. Explain below which split ratio worked best and why you think that is

**YOUR ANSWER:**
