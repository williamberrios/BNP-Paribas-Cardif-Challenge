<!-- #region -->
<p align="center">
    <img src="03.Resources/header.png"
     width="1200" height="120"/>
</p>

# Data Science Challenge BNP Paribas Cardif:
This is the first place solution to the machine learning competition BNP-Paribas
## Objective: 
[Competition](https://www.dominodatalab.com/blog/what-can-100-data-scientists-do-in-one-week-answer-a-lot) organized by [BNP Paribas Cardif](https://www.bnpparibascardif.com/en/) & [Domino Datalab](https://www.dominodatalab.com/) and consisted in predicting a food’s nutrient score based on its composition.

<p align="center">
    <img src="03.Resources/figure_nutrition.jpg"
    />
    <img src="03.Resources/nutrition_idx.gif" height = 295 width = 250
    />
</p>


## Participants: 
Around 100 students from Chile, Colombia, Peru and Mexico competed in the Data Science Challenge that was developed in Latin America.

## Data Available: 
The amount of carbohydrates, protein, vitamins , provenance, type of packaging, place of production, ingredients as text, etc. Download data [here](https://drive.google.com/drive/folders/1zkY4LemQTTp23WtrIVcA5II_zoUwFmV2?usp=sharing)

## Solution
All experiements we developed in Kfold validation. See folder 02.Code/ 
+  First we fine-tuned a [Roberta Transformer](https://arxiv.org/abs/1907.11692) only with concatenated text features.
+ After this we trained a lightgbm combining embeddings extracted from the transformer and numerical features.
+ Finally we ensemble kfold predictions for final submission.

<p align="center">
    <img src="03.Resources/BNP-Training-stages.png"/>
</p>
</p>
<p align = "center">
Fig. Training stages 1. Roberta Finetuning, 2. Kfold - Lightgbm training
</p>