# This script analyzes stroke occurrences versus 10 variables
# Several models are trained to predict strokes and assess stroke risk

# Install libraries if necessary
if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(varhandle))
  install.packages("varhandle", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(e1071))
  install.packages("e1071", repos = "http://cran.us.r-project.org")
if (!require(pROC))
  install.packages("pROC", repos = "http://cran.us.r-project.org")
if (!require(RANN))
  install.packages("RANN", repos = "http://cran.us.r-project.org")
if (!require(randomForest))
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
if (!require(ranger))
  install.packages("ranger", repos = "http://cran.us.r-project.org")
if (!require(rpart))
  install.packages("rpart", repos = "http://cran.us.r-project.org")
if (!require(gbm))
  install.packages("gbm", repos = "http://cran.us.r-project.org")
if (!require(nnet))
  install.packages("nnet", repos = "http://cran.us.r-project.org")

# Load libraries
library("tidyverse")
library("varhandle")
library("caret")
library("e1071")
library("pROC")
library("RANN")
library("randomForest")
library("ranger")
library("rpart")
library("gbm")
library("nnet")

# Download data set from kaggle
# https://www.kaggle.com/fedesoriano/stroke-prediction-dataset/download
# Unzip to extract csv file

# Read csv file into data frame
url <- "https://raw.githubusercontent.com/oneskychai/Stroke-Prediction/trunk/healthcare-dataset-stroke-data.csv"
dat <- read_csv(url)

# Create rdas directory if doesn't exist and save dat
wd <- getwd()
ifelse(!dir.exists(file.path(wd, "rdas")),
       dir.create(file.path(wd, "rdas")),
       FALSE)
save(dat, file = "rdas/stroke_data.rda")

################
# Explore data #
################

# Examine structure of dat
str(dat)

# Remove id column, convert bmi to numeric, convert stroke to factor
dat <- dat %>%
  select(-id) %>%
  mutate(bmi = as.numeric(bmi)) %>%
  mutate(stroke = as.factor(stroke))

# Save cleaned version of dat
save(dat, file = "rdas/dat_clean.rda")

# Count total NA's
colSums(is.na(dat)) %>% knitr::kable()

#================================#
# Examine variable distributions #
#================================#

# Examine gender distribution
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(gender) %>%
  summarize(count = n(),
            percentage = round(100 * count / nrow(dat), 2),
            ypos = round(count / 2))

# Round female and male percentages to 1 digit
info[1:2, 3] <- round(info[1:2, 3], 1)

# Adjust other gender label position
info[3, 4] <- 100
info

# Plot gender distribution
info %>%
  mutate(gender = reorder(gender, count)) %>%
  ggplot(aes(gender, count, fill = gender)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.8,
           show.legend = FALSE) +
  scale_fill_manual(values = c("seagreen3", "skyblue3", "orchid3")) +
  geom_text(aes(label = paste0(percentage, "%"), y = ypos)) +
  ggtitle("Gender distribution")

# Save plot
ifelse(!dir.exists(file.path(wd, "figs")),
       dir.create(file.path(wd, "figs")),
       FALSE)
rm(wd)
ggsave("figs/gender_distribution.png", dpi = 95)

# Plot age distribution
dat %>%
  ggplot(aes(age)) +
  geom_histogram(color = "black",
                 fill = "slateblue3",
                 breaks = seq(0, 85, 5)) +
  ggtitle("Age distribution")

# Save plot
ggsave("figs/age_distribution.png", dpi = 95)

# Display age range
range(dat$age)

# Examine hypertension distribution
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(hypertension) %>%
  summarize(count = n(),
            percentage = round(100 * count / nrow(dat), 1),
            ypos = round(count / 2))
info

# Plot hypertension distribution
info %>%
  mutate(hypertension = reorder(hypertension, count)) %>%
  ggplot(aes(hypertension, count, fill = hypertension)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("brown2", "royalblue3")) +
  geom_text(aes(label = paste0(percentage, "%"), y = ypos)) +
  scale_x_discrete(labels = c("yes", "no")) +
  ggtitle("Hypertension distribution")

# Save plot
ggsave("figs/hypertension_distribution.png", dpi = 95)

# Examine heart disease distribution
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(heart_disease) %>%
  summarize(count = n(),
            percentage = round(100 * count / nrow(dat), 1),
            ypos = round(count / 2))

# Adjust heart disease yes label position
info[2, 4] <- 460
info

# Plot heart disease distribution
info %>%
  mutate(heart_disease = reorder(heart_disease, count)) %>%
  ggplot(aes(heart_disease, count, fill = heart_disease)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("brown2", "royalblue3")) +
  geom_text(aes(label = paste0(percentage, "%"), y = ypos)) +
  scale_x_discrete(labels = c("yes", "no")) +
  xlab("heart disease") +
  ggtitle("Heart disease distribution")

# Save plot
ggsave("figs/heart_disease_distribution.png", dpi = 95)

# Examine marriage status distribution
# Create ypos column for plot label positioning
info  <- dat %>%
  group_by(ever_married) %>%
  summarize(count = n(),
            percentage = round(100 * count / nrow(dat), 1),
            ypos = round(count / 2))
info

# Plot marriage status distribution
info %>%
  ggplot(aes(ever_married, count, fill = ever_married)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("seagreen3", "orchid3")) +
  geom_text(aes(label = paste0(percentage, "%"), y = ypos)) +
  scale_x_discrete(labels = c("no", "yes")) +
  xlab("ever married") +
  ggtitle("Marriage status distribution")

# Save plot
ggsave("figs/marriage_status_distribution.png", dpi = 95)

# Examine work type distribution
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(work_type) %>%
  summarize(count = n(),
            percentage = round(100 * count / nrow(dat), 1),
            ypos = round(count / 2))

# Adjust never worked label position
info[3, 4] <- 130
info

# Plot work type distribution
info %>%
  mutate(work_type = reorder(work_type, count)) %>%
  ggplot(aes(work_type, count, fill = work_type)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.8,
           show.legend = FALSE) +
  scale_fill_manual(values = c("orchid3", "slateblue3", "seagreen3",
                               "chocolate2", "firebrick2")) +
  geom_text(aes(label = paste0(percentage, "%"), y = ypos)) +
  xlab("work type") +
  scale_x_discrete(labels = c("never worked", "govt job", "children",
                              "self-employed", "private")) +
  ggtitle("Work type distribution")

# Save plot
ggsave("figs/work_type_distribution.png", dpi = 95)

# Examine residence type distribution
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(Residence_type) %>%
  summarize(count = n(),
            percentage = round(100 * count / nrow(dat), 1),
            ypos = round(count / 2))
info

# Plot residence type distribution
info %>%
  ggplot(aes(Residence_type, count, fill = Residence_type)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("tan4", "steelblue3")) +
  geom_text(aes(label = paste0(percentage, "%"), y = ypos)) +
  scale_x_discrete(labels = c("rural", "urban")) +
  xlab("residence type") +
  ggtitle("Residence type distribution")

# Save plot
ggsave("figs/residence_type_distribution.png", dpi = 95)

# Plot average glucose level distribution
dat %>%
  ggplot(aes(avg_glucose_level)) +
  geom_histogram(color = "black",
                 fill = "orchid3") +
  xlab("average glucose level") +
  ggtitle("Average glucose level distribution")

# Save plot
ggsave("figs/avg_glucose_distribution.png", dpi = 95)

# Plot bmi distribution
dat %>%
  ggplot(aes(bmi)) +
  geom_histogram(color = "black",
                 fill = "seagreen3") +
  xlab("body mass index") +
  ggtitle("Body mass index distribution")

# Save plot
ggsave("figs/bmi_distribution.png", dpi = 95)

# Examine smoking status distribution
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(smoking_status) %>%
  summarize(count = n(),
            percentage = round(100 * count / nrow(dat), 1),
            ypos = round(count / 2))
info

# Plot smoking status distribution
info %>%
  mutate(smoking_status = reorder(smoking_status, count)) %>%
  ggplot(aes(smoking_status, count, fill = smoking_status)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.7,
           show.legend = FALSE) +
  scale_fill_manual(values = c("firebrick3", "orchid3",
                               "slateblue3", "seagreen3")) +
  geom_text(aes(label = paste0(percentage, "%"), y = ypos)) +
  xlab("smoking status") +
  ggtitle("Smoking status distribution")

# Save plot
ggsave("figs/smoking_status_distribution.png", dpi = 95)

# Examine stroke occurrence distribution
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(stroke) %>%
  summarize(count = n(),
            percentage = round(100 * count / nrow(dat), 1),
            ypos = round(count / 2))

# Adjust stroke yes label position
info[2, 4] <- 430
info

# Plot stroke event distribution
info %>%
  mutate(stroke = reorder(stroke, count)) %>%
  ggplot(aes(stroke, count, fill = stroke)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("brown2", "royalblue3")) +
  geom_text(aes(label = paste0(percentage, "%"), y = ypos)) +
  scale_x_discrete(labels = c("yes", "no")) +
  xlab("stroke event") +
  ggtitle("Stroke event distribution")

# Save plot
ggsave("figs/stroke_distribution.png", dpi = 95)

#==========================================================#
# Examine correlations between variables and stroke events #
#==========================================================#

# Examine stroke rate by gender
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(gender) %>%
  summarize(n = n(),
            strokes = sum(stroke == 1),
            stroke_percent = round(100 * strokes / n, 1),
            ypos = stroke_percent / 2)

# Adjust other gender label position
info[3, 5] <- 0.2
info

# Plot stroke rate by gender
info %>%
  mutate(gender = reorder(gender, stroke_percent)) %>%
  ggplot(aes(gender, stroke_percent, fill = gender)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.8,
           show.legend = FALSE) +
  scale_fill_manual(values = c("seagreen3", "orchid3", "skyblue3")) +
  geom_text(aes(label = paste0(stroke_percent, "%"), y = ypos)) +
  scale_y_continuous(breaks = c(0, 2, 4),
                     labels = paste0(c(0, 2, 4), "%")) +
  ylab("stroke rate") +
  ggtitle("Stroke rate by gender")

# Save plot
ggsave("figs/strokes_by_gender.png", dpi = 95)

# Examine age distribution by stroke occurrence
dat %>%
  filter(stroke == 0) %>%
  .$age %>%
  summary()

dat %>%
  filter(stroke == 1) %>%
  .$age %>%
  summary()

# Boxplot age distribution by stroke occurrence
dat %>%
  ggplot(aes(stroke, age)) +
  geom_boxplot(color = "black",
               fill = "slateblue3") +
  xlab("stroke event") +
  scale_x_discrete(labels = c("no", "yes")) +
  coord_flip() +
  ggtitle("Age distribution by stroke occurrence")

# Save plot
ggsave("figs/age_distribution_by_stroke_boxplot.png", dpi = 95)

# Histogram plot age distribution by stroke occurrence
# Change labels on facet grid from "0" and "1" to "stroke" and "no stroke"
stroke.labs <- c("no stroke", "stroke")
names(stroke.labs) <- c(0, 1)

dat %>%
  mutate(stroke = reorder(stroke, desc(stroke))) %>%
  ggplot(aes(age)) +
  geom_histogram(color = "black",
                 fill = "slateblue3",
                 breaks = seq(0, 85, 5)) +
  facet_grid(stroke ~ .,
             labeller = labeller(stroke = stroke.labs)) +
  ggtitle("Age distribution by stroke occurrence")

# Save plot
ggsave("figs/age_distribution_by_stroke_histogram.png", dpi = 95)

# Plot stroke rate versus age stratified every 5 years
dat %>%
  mutate(age_strata = ceiling(age / 5)) %>%
  group_by(age_strata) %>%
  summarize(stroke_percent = 100 * sum(stroke == 1) / n()) %>%
  ggplot(aes(age_strata, stroke_percent)) +
  geom_smooth(color = "slateblue3",
              size = 1,
              se = FALSE) +
  geom_point(color = "black",
             fill = "slateblue3",
             size = 3,
             shape = 21,
             alpha = 0.5) +
  xlab("age group") +
  scale_x_continuous(breaks = seq(1, 16, 3),
                     labels = c("0-5", "15-20", "30-35",
                                "45-50", "60-65", "75-80")) +
  ylab("stroke rate") +
  scale_y_continuous(breaks = seq(0, 20, 5),
                     labels = paste0(seq(0, 20, 5), "%")) +
  ggtitle("Stroke rate by age group")

# Save plot
ggsave("figs/stroke_rate_by_age.png", dpi = 95)

# Examine stroke rate by hypertension
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(hypertension) %>%
  summarize(n = n(),
            strokes = sum(stroke == 1),
            stroke_percent = round(100 * strokes / n, 1),
            ypos = stroke_percent / 2)
info

# Plot stroke rate by hypertension
info %>%
  mutate(hypertension = as.character(hypertension)) %>%
  ggplot(aes(hypertension, stroke_percent, fill = hypertension)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("royalblue3", "brown2")) +
  geom_text(aes(label = paste0(stroke_percent, "%"),
                y = ypos)) +
  scale_x_discrete(labels = c("no", "yes")) +
  ylab("stroke rate") +
  scale_y_continuous(breaks = c(0, 5, 10),
                     labels = paste0(c(0, 5, 10), "%")) +
  ggtitle("Stroke rate by hypertension")

# Save plot
ggsave("figs/strokes_by_hypertension.png", dpi = 95)

# Plot age distribution by hypertension
dat %>%
  mutate(hypertension = as.factor(hypertension)) %>%
  ggplot(aes(hypertension, age, fill = hypertension)) +
  geom_boxplot(color = "black",
               show.legend = FALSE) +
  scale_fill_manual(values = c("royalblue3", "brown2")) +
  xlab("hypertension") +
  scale_x_discrete(labels = c("no", "yes")) +
  ylab("age") +
  coord_flip() +
  ggtitle("Age distribution by hypertension")

# Save plot
ggsave("figs/age_distribution_by_hypertension.png", dpi = 95)

# Calculate correlation between age and hypertension
cor(dat$age, dat$hypertension)

# Examine stroke rate by heart disease
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(heart_disease) %>%
  summarize(n = n(),
            strokes = sum(stroke == 1),
            stroke_percent = round(100 * strokes / n, 1),
            ypos = stroke_percent / 2)
info

# Plot stroke rate by heart disease
info %>%
  mutate(heart_disease = as.character(heart_disease)) %>%
  ggplot(aes(heart_disease, stroke_percent, fill = heart_disease)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("royalblue3", "brown2")) +
  geom_text(aes(label = paste0(stroke_percent, "%"), y = ypos)) +
  xlab("heart disease") +
  scale_x_discrete(labels = c("no", "yes")) +
  ylab("stroke rate") +
  scale_y_continuous(breaks = seq(0, 15, 5),
                     labels = paste0(seq(0, 15, 5), "%")) +
  ggtitle("Stroke rate by heart disease")

# Save plot
ggsave("figs/strokes_by_heart_disease.png", dpi = 95)

# Plot age distribution by heart disease
dat %>%
  mutate(heart_disease = as.factor(heart_disease)) %>%
  ggplot(aes(heart_disease, age, fill = heart_disease)) +
  geom_boxplot(color = "black",
               show.legend = FALSE) +
  scale_fill_manual(values = c("royalblue3", "brown2")) +
  xlab("heart disease") +
  scale_x_discrete(labels = c("no", "yes")) +
  ylab("age") +
  coord_flip() +
  ggtitle("Age distribution by heart disease")

# Save plot
ggsave("figs/age_distribution_by_heart_disease.png", dpi = 95)

# Calculate correlation between age and heart disease
cor(dat$age, dat$heart_disease)

# Examine stroke rate by marriage status
# Create ypos column for plot label positioning
info<- dat %>%
  group_by(ever_married) %>%
  summarize(n = n(),
            strokes = sum(stroke == 1),
            stroke_percent = round(100 * strokes / n, 1),
            ypos = stroke_percent / 2)
info

# Plot stroke rate by marriage status
info %>%
  ggplot(aes(ever_married, stroke_percent, fill = ever_married)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("seagreen3", "orchid3")) +
  geom_text(aes(label = paste0(stroke_percent, "%"), y = ypos)) +
  xlab("ever married") +
  scale_x_discrete(labels = c("no", "yes")) +
  ylab("stroke rate") +
  scale_y_continuous(labels = paste0(seq(0, 6, 2), "%")) +
  ggtitle("Stroke rate by marriage status")

# Save plot
ggsave("figs/strokes_by_marriage_status.png", dpi = 95)

# Plot age distribution by marriage status
dat %>%
  ggplot(aes(ever_married, age, fill = ever_married)) +
  geom_boxplot(color = "black",
               show.legend = FALSE) +
  scale_fill_manual(values = c("seagreen3", "orchid3")) +
  xlab("ever married") +
  ylab("age") +
  coord_flip() +
  ggtitle("Age distribution by marriage status")

# Save plot
ggsave("figs/age_distribution_by_marriage.png", dpi = 95)

# Calculate correlation between age and marriage status
dat %>%
  mutate(ever_married = as.factor(ever_married)) %>%
  summarize(cor(age, as.numeric(ever_married)))

# Examine stroke rate by work type
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(work_type) %>%
  summarize(n = n(),
            strokes = sum(stroke == 1),
            stroke_percent = round(100 * strokes / n, 1),
            ypos = stroke_percent / 2)

# Adjust never worked and children label positions
info[3, 5] <- 0.3
info[1, 5] <- 0.6
info

# Plot stroke rate by work type
info %>%
  mutate(work_type = reorder(work_type, stroke_percent)) %>%
  ggplot(aes(work_type, stroke_percent, fill = work_type)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.8,
           show.legend = FALSE) +
  scale_fill_manual(values = c("orchid3", "seagreen3", "slateblue3",
                               "firebrick2", "chocolate2")) +
  geom_text(aes(label = paste0(stroke_percent, "%"), y = ypos)) +
  scale_x_discrete(labels = c("never worked", "children", "govt job",
                              "private", "self-employed")) +
  xlab("work type") +
  scale_y_continuous(breaks = seq(0, 8, 2),
                     labels = paste0(seq(0, 8, 2), "%")) +
  ylab("stroke rate") +
  ggtitle("Stroke rate by work type")

# Save plot
ggsave("figs/strokes_by_work_type.png", dpi = 95)

# Plot age distribution by work type
dat %>%
  mutate(work_type = as.factor(work_type)) %>%
  mutate(work_type = fct_reorder(work_type, age)) %>%
  ggplot(aes(work_type, age, fill = work_type)) +
  geom_boxplot(color = "black",
               show.legend = FALSE) +
  scale_fill_manual(values = c("seagreen3", "orchid3", "firebrick2",
                               "slateblue3", "chocolate2")) +
  xlab("work type") +
  ylab("age") +
  scale_x_discrete(labels = c("children", "never worked", "private",
                              "govt job", "self-employed")) +
  coord_flip() +
  ggtitle("Age distribution by work type")

# Save plot
ggsave("figs/age_distribution_by_work_type.png", dpi = 95)

# Examine stroke rate by residence type
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(Residence_type) %>%
  summarize(n = n(),
            strokes = sum(stroke == 1),
            stroke_percent = round(100 * strokes / n, 1),
            ypos = stroke_percent / 2)
info

# Plot stroke rate by residence type
info %>%
  ggplot(aes(Residence_type, stroke_percent, fill = Residence_type)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.6,
           show.legend = FALSE) +
  scale_fill_manual(values = c("tan4", "steelblue3")) +
  geom_text(aes(label = paste0(stroke_percent, "%"), y = ypos)) +
  xlab("residence type") +
  scale_y_continuous(breaks = c(0, 2, 4),
                     labels = paste0(c(0, 2, 4), "%")) +
  ylab("stroke rate") +
  ggtitle("Stroke rate by residence type")

# Save plot
ggsave("figs/strokes_by_residence_type.png", dpi = 95)

# Plot age distribution by residence type
dat %>%
  ggplot(aes(Residence_type, age, fill = Residence_type)) +
  geom_boxplot(color = "black",
               show.legend = FALSE) +
  scale_fill_manual(values = c("tan4", "steelblue3")) +
  xlab("residence type") +
  ylab("age") +
  coord_flip() +
  ggtitle("Age distribution by residence type")

# Save plot
ggsave("figs/age_distribution_by_residence_type.png", dpi = 95)

# Calculate correlation between age and residence type
dat %>%
  mutate(Residence_type = as.factor(Residence_type)) %>%
  summarize(cor(age, as.numeric(Residence_type)))

# Examine average glucose level distribution by stroke occurrence
dat %>%
  filter(stroke == 0) %>%
  .$avg_glucose_level %>%
  summary()

dat %>%
  filter(stroke == 1) %>%
  .$avg_glucose_level %>%
  summary()

# Boxplot average glucose level distribution by stroke occurrence
dat %>%
  ggplot(aes(stroke, avg_glucose_level)) +
  geom_boxplot(color = "black",
               fill = "orchid3") +
  scale_x_discrete(labels = c("no", "yes")) +
  xlab("stroke event") +
  ylab("average glucose level") +
  coord_flip() +
  ggtitle("Average glucose level distribution by stroke occurrence")

# Save plot
ggsave("figs/glucose_distribution_by_stroke_boxplot.png", dpi = 95)

# Histogram plot average glucose level distribution by stroke occurrence
dat %>%
  mutate(stroke = reorder(stroke, desc(stroke))) %>%
  ggplot(aes(avg_glucose_level)) +
  geom_histogram(color = "black",
                 fill = "orchid3") +
  xlab("average glucose level") +
  facet_grid(stroke ~ .,
             labeller = labeller(stroke = stroke.labs)) +
  ggtitle("Average glucose level distribution by stroke occurrence")

# Save plot
ggsave("figs/glucose_distribution_by_stroke_histogram.png", dpi = 95)

# Plot stroke rate versus average glucose level stratified by every 20 points
dat %>%
  mutate(agl_strata = ceiling(avg_glucose_level / 20)) %>%
  group_by(agl_strata) %>%
  summarize(stroke_percent = 100 * sum(stroke == 1) / n()) %>%
  ggplot(aes(agl_strata, stroke_percent)) +
  geom_smooth(color = "orchid3",
              size = 1,
              se = FALSE) +
  geom_point(color = "black",
             fill = "orchid3",
             shape = 21,
             size = 3,
             alpha = 0.5) +
  xlab("average glucose level") +
  scale_x_continuous(breaks = seq(3.5, 13.5, 2),
                     labels = seq(60, 260, 40)) +
  ylab("stroke rate") +
  scale_y_continuous(limits = c(0, 22.3),
                     breaks = seq(0, 20, 5),
                     labels = paste0(seq(0, 20, 5), "%")) +
  ggtitle("Stroke rate by average glucose level")

# Save plot
ggsave("figs/stroke_rate_by_glucose.png", dpi = 95)

# Calculate correlation between age and average glucose level
cor(dat$age, dat$avg_glucose_level)

# Examine bmi distribution by stroke occurrence
dat %>%
  filter(stroke == 0) %>%
  .$bmi %>%
  summary()

dat %>%
  filter(stroke == 1) %>%
  .$bmi %>%
  summary()

# Boxplot bmi distribution by stroke occurrence
dat %>%
  ggplot(aes(stroke, bmi)) +
  geom_boxplot(color = "black",
               fill = "seagreen3") +
  scale_x_discrete(labels = c("no", "yes")) +
  xlab("stroke event") +
  ylab("body mass index") +
  coord_flip() +
  ggtitle("Body mass index distribution by stroke occurrence")

# Save plot
ggsave("figs/bmi_distribution_by_stroke_boxplot.png", dpi = 95)

# Histogram plot bmi distribution by stroke occurrence
dat %>%
  mutate(stroke = reorder(stroke, desc(stroke))) %>%
  ggplot(aes(bmi)) +
  geom_histogram(color = "black",
                 fill = "seagreen3") +
  xlab("body mass index") +
  facet_grid(stroke ~ .,
             labeller = labeller(stroke = stroke.labs)) +
  ggtitle("Body mass index distribution by stroke occurrence")
rm(stroke.labs)

# Save plot
ggsave("figs/bmi_distribution_by_stroke_histogram.png", dpi = 95)

# Plot stroke rate versus bmi stratified by every 3 points
dat %>%
  filter(!is.na(bmi)) %>%
  mutate(bmi_strata = ceiling(bmi / 3)) %>%
  # Consolidate tails of bmi strata to reduce outlier distortion
  mutate(bmi_strata = case_when(bmi_strata < 7 ~ 6,
                                bmi_strata > 15 ~ 16,
                                TRUE ~ bmi_strata)) %>%
  group_by(bmi_strata) %>%
  summarize(stroke_percent = 100 * sum(stroke == 1) / n()) %>%
  ggplot(aes(bmi_strata, stroke_percent)) +
  geom_smooth(color = "seagreen3",
              size = 1,
              se = FALSE) +
  geom_point(color = "black",
             fill = "seagreen3",
             size = 3,
             shape = 21,
             alpha = 0.5) +
  xlab("body mass index") +
  scale_x_continuous(breaks = seq(7.167, 15.5, 1.6666),
                     labels = c(seq(20, 40, 5), "45+")) +
  ylab("stroke rate") +
  scale_y_continuous(breaks = seq(0, 6, 2),
                     labels = paste0(seq(0, 6, 2), "%")) +
  ggtitle("Stroke rate by body mass index")

# Save plot
ggsave("figs/stroke_rate_by_bmi.png", dpi = 95)

# Calculate correlation between age and bmi
dat %>%
  filter(!is.na(bmi)) %>%
  summarize(cor(age, bmi))

# Examine stroke rate by smoking status
# Create ypos column for plot label positioning
info <- dat %>%
  group_by(smoking_status) %>%
  summarize(n = n(),
            strokes = sum(stroke == 1),
            stroke_percent = round(100 * strokes / n, 1),
            ypos = stroke_percent / 2)
info

# Plot stroke rate by smoking status
info %>%
  mutate(smoking_status = reorder(smoking_status, stroke_percent)) %>%
  ggplot(aes(smoking_status, stroke_percent, fill = smoking_status)) +
  geom_bar(stat = "identity",
           color = "black",
           width = 0.7,
           show.legend = FALSE) +
  scale_fill_manual(values = c("slateblue3", "seagreen3",
                               "firebrick3", "orchid3")) +
  geom_text(aes(label = paste0(stroke_percent, "%"), y = ypos)) +
  xlab("smoking status") +
  ylab("stroke rate") +
  scale_y_continuous(breaks = seq(0, 8, 2),
                     labels = paste0(seq(0, 8, 2), "%")) +
  ggtitle("Stroke rate by smoking status")

# Save plot
ggsave("figs/strokes_by_smoking_status.png", dpi = 95)

# Compare age distribution by smoking status
dat %>%
  mutate(smoking_status = as.factor(smoking_status)) %>%
  mutate(smoking_status = fct_reorder(smoking_status, age)) %>%
  ggplot(aes(smoking_status, age, fill = smoking_status)) +
  geom_boxplot(color = "black",
               show.legend = FALSE) +
  scale_fill_manual(values = c("slateblue3", "seagreen3",
                               "firebrick3", "orchid3")) +
  xlab("smoking status") +
  ylab("age") +
  coord_flip() +
  ggtitle("Age distribution by smoking status")

# Save plot
ggsave("figs/age_distribution_by_smoking_status.png", dpi = 95)

#######################################
# Train models for predicting strokes #
#######################################

#================#
# Preprocess dat #
#================#

# Count number of NA's in bmi column where stroke = 1
dat %>%
  filter(is.na(bmi)) %>%
  summarize(sum(stroke == 1))

# Calculate percentage of stroke cases where bmi is NA
dat %>%
  filter(stroke == 1) %>%
  summarize(percent = 100 * mean(is.na(bmi))) %>%
  .$percent

# Examine data where bmi is NA
bmi_na <- dat %>%
  filter(is.na(bmi))
view(bmi_na)
summary(bmi_na$age)
summary(dat$age)

# Impute NA values in bmi column
# Note setting the seed here does not guarantee reproducible results
set.seed(432, sample.kind = "Rounding")
values <- preProcess(as.data.frame(dat), method = "bagImpute")
dat <- predict(values, dat)

# Set gender = "Male" for case where gender = "Other"
ind <- which(dat$gender == "Other")
dat$gender[ind] <- "Male"

# Convert categorical variables to factors
dat <- dat %>%
  mutate(gender = as.factor(gender),
         hypertension = as.factor(hypertension),
         heart_disease = as.factor(heart_disease),
         ever_married = as.factor(ever_married),
         work_type = as.factor(work_type),
         Residence_type = as.factor(Residence_type),
         smoking_status = as.factor(smoking_status))

# Save processed version of dat
save(dat, file = "rdas/dat_processed.rda")

#----------------------------------#
# Create stratified version of dat #
#----------------------------------#

# Stratify age by every 5 years
strat <- dat %>%
  mutate(age = ceiling(age / 5))

# Examine distribution of age strata
table(strat$age)

# Stratify avg_glucose_level by every 20 points
strat <- strat %>%
  mutate(avg_glucose_level = ceiling(avg_glucose_level / 20))

# Examine distribution of avg_glucose_level strata
table(strat$avg_glucose_level)

# Consolidate tail end of avg_glucose_level strata
strat <- strat %>%
  mutate(avg_glucose_level = ifelse(avg_glucose_level > 11,
                                    12, avg_glucose_level))

# Stratify bmi by every 3 points
strat <- strat %>%
  mutate(bmi = ceiling(bmi / 3))

# Examine distribution of bmi strata
table(strat$bmi)

# Consolidate tail ends of bmi strata
strat <- strat %>%
  mutate(bmi = case_when(bmi < 7 ~ 6,
                         bmi > 15 ~ 16,
                         TRUE ~ bmi))

# Set strat$stroke levels to no and yes
levels(strat$stroke) <- c("no", "yes")

#----------------------------------------------------------#
# Employ one hot encoding on dat for categorical variables #
#----------------------------------------------------------#

# Convert categorical variables to numeric with dummy variables
# Convert stroke to numeric first so it is unaffected
dat$stroke <- unfactor(dat$stroke)
dummy <- dummyVars( ~ ., data = dat, fullRank = TRUE)
dat <- data.frame(predict(dummy, dat))
rm(dummy)

# Convert stroke to factor with levels no and yes
dat$stroke <- as.factor(dat$stroke)
levels(dat$stroke) <- c("no", "yes")

# Convert dat back to tibble
dat <- as_tibble(dat)

#========================================#
# Split data into training and test sets #
#========================================#

# Split data into train and test sets
set.seed(23, sample.kind = "Rounding")
test_index <- createDataPartition(dat$stroke, times = 1, p = 0.1, list = FALSE)
test_index <- as.vector(test_index)
test <- dat[test_index,]
train <- dat[-test_index,]
test_strat <- strat[test_index,]
train_strat <- strat[-test_index,]

#==========#
# rf model #
#==========#

# Set up train control function
ctrl <- trainControl(method = "repeatedcv",
                     number = 5,
                     repeats = 3,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     verboseIter = TRUE)

# Train random forest model
# Note this will take a few minutes
set.seed(316, sample.kind = "Rounding")
train_rf <- train(stroke ~ .,
                  method = "rf",
                  data = train,
                  metric = "ROC",
                  trControl = ctrl)
save(train_rf, file = "rdas/train_rf.rda")

# Examine variable importance
varImp(train_rf)

# Make predictions on test set with rf model
pred_rf <- predict(train_rf, test)

# Examine confusion matrix of results
cm <- confusionMatrix(pred_rf, test$stroke, positive = "yes")
cm

# Save results to data frame
acc <- unname(cm$byClass[11])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- data.frame(model = "rf",
                               balanced_accuracy = acc,
                               sensitivity = sens,
                               specificity = spec)

# Extract ROC data from model
test_roc <- roc(test$stroke,
                predict(train_rf, test, type = "prob")[, "yes"])

# Calculate AUC
auc(test_roc)

# Save AUC to dataframe
auc_results <- data.frame(model = "rf",
                          AUC = auc(test_roc))

# Pull true positive and false positive rates from test_roc
# Reverse order of vectors for cleaner lines when plotting
df_roc <- data.frame(tpr = rev(test_roc$sensitivities),
                     fpr = rev(1 - test_roc$specificities),
                     model = "rf")

# Plot ROC curve
df_roc %>%
  ggplot(aes(fpr, tpr)) +
  geom_line(color = "navy",
            size = 1) +
  geom_abline(color = "gray") +
  geom_abline(intercept = 1,
              slope = -1,
              color = "gray") +
  xlab("false positive rate") +
  ylab("true positive rate") +
  ggtitle("ROC curve for rf model")

# Save plot
ggsave("figs/roc_rf.png", dpi = 95)

# Predict probabilities on test set with rf model
probs_rf <- predict(train_rf, test, type = "prob")$yes

# Examine probability distributions for yes when stroke = yes and stroke = no
summary(probs_rf[test$stroke == "yes"])
summary(probs_rf[test$stroke == "no"])

# Make predictions with a threshold of 0.07 for yes
pred_rf_07 <- as.factor(ifelse(probs_rf > 0.07, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_rf_07, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("rf_07", acc, sens, spec))

#===========================#
# ranger model with weights #
#===========================#

# Create weights to account for imbalanced class distribution
model_weights <- ifelse(train$stroke == "no",
                        (1/table(train$stroke)[1]) * 0.5,
                        (1/table(train$stroke)[2]) * 0.5)
sum(model_weights)

# Set up tunegrid
tune <- expand.grid(mtry = 5:8,
                    splitrule = c("gini", "extratrees"),
                    min.node.size = 1)

# Train weighted ranger model
# Use same tune control function from last model
# Note this will take a few minutes
set.seed(1714, sample.kind = "Rounding")
train_ranger <- train(stroke ~ .,
                      method = "ranger",
                      data = train,
                      weights = model_weights,
                      importance = "permutation",
                      metric = "ROC",
                      trControl = ctrl,
                      tuneGrid = tune)
save(train_ranger, file = "rdas/train_ranger.rda")

# Examine variable importance
varImp(train_ranger)

# Make predictions on test set with weighted ranger model
pred_ranger <- predict(train_ranger, test)

# Examine confusion matrix of results
cm <- confusionMatrix(pred_ranger, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("ranger", acc, sens, spec))

# Extract ROC data from model
test_roc <- roc(test$stroke,
                predict(train_ranger, test, type = "prob")[, "yes"])

# Calculate AUC
auc(test_roc)

# Add AUC to auc_results
auc_results <- rbind(auc_results, c("ranger", auc(test_roc)))

# Pull true positive and false positive rates from test_roc
df_roc_ranger <- data.frame(tpr = rev(test_roc$sensitivities),
                            fpr = rev(1 - test_roc$specificities),
                            model = "ranger")

# Add ROC data from ranger model to df_roc
df_roc <- rbind(df_roc, df_roc_ranger)

# Compare ROC curves from rf and ranger models
df_roc %>%
  ggplot(aes(fpr, tpr, color = model)) +
  geom_line(size = 1) +
  geom_abline(color = "gray") +
  geom_abline(intercept = 1,
              slope = -1,
              color = "gray") +
  xlab("false positive rate") +
  ylab("true positive rate") +
  ggtitle("ROC curve for rf and ranger models")

# Save plot
ggsave("figs/roc_rf_ranger.png", dpi = 95)

# Predict probabilities on test set with weighted ranger model
probs_ranger <- predict(train_ranger, test, type = "prob")$yes

# Examine probability distributions for yes when stroke = yes and stroke = no
summary(probs_ranger[test$stroke == "yes"])
summary(probs_ranger[test$stroke == "no"])

# Make predictions with a threshold of 0.16 for yes
pred_ranger_16 <- as.factor(ifelse(probs_ranger > 0.16, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_ranger_16, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("ranger_16", acc, sens, spec))

# Make predictions with a threshold of 0.12 for yes
pred_ranger_12 <- as.factor(ifelse(probs_ranger > 0.12, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_ranger_12, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("ranger_12", acc, sens, spec))

#==========================#
# rpart model with weights #
#==========================#

# Adjust train control function
ctrl$number <- 10
ctrl$repeats <- 5

# Train weighted rpart model
set.seed(1350, sample.kind = "Rounding")
train_rpart <- train(stroke ~ .,
                     method = "rpart",
                     data = train,
                     weights = model_weights,
                     metric = "ROC",
                     tuneGrid = data.frame(cp = seq(0.008, 0.0086, 0.00001)),
                     trControl = ctrl)

# Save model
save(train_rpart, file = "rdas/train_rpart.rda")

# Examine variable importance
varImp(train_rpart)

# Examine decision tree
plot(train_rpart$finalModel, margin = 0.05)
text(train_rpart$finalModel, cex = 0.75)

# Make predictions on test set with weighted rpart model
pred_rpart <- predict(train_rpart, test)

# Examine confusion matrix of results
cm <- confusionMatrix(pred_rpart, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("rpart", acc, sens, spec))

# Extract ROC data from model
test_roc <- roc(test$stroke,
                predict(train_rpart, test, type = "prob")[, "yes"])

# Calculate AUC
auc(test_roc)

# Add AUC to auc_results
auc_results <- rbind(auc_results, c("rpart", auc(test_roc)))

# Pull true positive and false positive rates from test_roc
df_roc_rpart <- data.frame(tpr = rev(test_roc$sensitivities),
                           fpr = rev(1 - test_roc$specificities),
                           model = "rpart")

# Add ROC data from rpart model to df_roc
df_roc <- rbind(df_roc, df_roc_rpart)

# Compare ROC curves from different models
df_roc %>%
  ggplot(aes(fpr, tpr, color = model)) +
  geom_line(size = 1) +
  geom_abline(color = "gray") +
  geom_abline(intercept = 1,
              slope = -1,
              color = "gray") +
  xlab("false positive rate") +
  ylab("true positive rate") +
  ggtitle("ROC curve by model")

# Save plot
ggsave("figs/roc_models_3.png", dpi = 95)

# Predict probabilities on test set with weighted rpart model
probs_rpart <- predict(train_rpart, test, type = "prob")$yes

# Examine probability distributions for yes when stroke = yes and stroke = no
summary(probs_rpart[test$stroke == "yes"])
summary(probs_rpart[test$stroke == "no"])

# Inspect table of probabilities from rpart model
table(probs_rpart)

#========================#
# gbm model with weights #
#========================#

# Set up tunegrid
tune <- expand.grid(interaction.depth = c(1, 2, 3),
                    n.trees = c(50, 100, 150),
                    shrinkage = c(0.01, 0.05, 0.01),
                    n.minobsinnode = 10)

# Train weighted gbm model
# Use same train control function from last model
# Note this will take a few minutes
set.seed(2021, sample.kind = "Rounding")
train_gbm <- train(stroke ~ .,
                   method = "gbm",
                   data = train,
                   weights = model_weights,
                   nTrain = round(nrow(train) * 0.8),
                   metric = "ROC",
                   trControl = ctrl,
                   tuneGrid = tune)

# Save model
save(train_gbm, file = "rdas/train_gbm.rda")

# Examine variable importance
varImp(train_gbm)

# Make predictions on test set with weighted gbm model
pred_gbm <- predict(train_gbm, test)

# Examine confusion matrix of results
cm <- confusionMatrix(pred_gbm, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("gbm", acc, sens, spec))

# Extract ROC data from model
test_roc <- roc(test$stroke,
                predict(train_gbm, test, type = "prob")[, "yes"])

# Calculate AUC
auc(test_roc)

# Add AUC to auc_results
auc_results <- rbind(auc_results, c("gbm", auc(test_roc)))

# Pull true positive and false positive rates from test_roc
df_roc_gbm <- data.frame(tpr = rev(test_roc$sensitivities),
                         fpr = rev(1 - test_roc$specificities),
                         model = "gbm")

# Add ROC data from gbm model to df_roc
df_roc <- rbind(df_roc, df_roc_gbm)

# Compare ROC curves from different models
df_roc %>%
  ggplot(aes(fpr, tpr, color = model)) +
  geom_line(size = 1) +
  geom_abline(color = "gray") +
  geom_abline(intercept = 1,
              slope = -1,
              color = "gray") +
  xlab("false positive rate") +
  ylab("true positive rate") +
  ggtitle("ROC curve by model")

# Save plot
ggsave("figs/roc_models_4.png", dpi = 95)

# Predict probabilities on test set with weighted gbm model
probs_gbm <- predict(train_gbm, test, type = "prob")$yes

# Examine probability distributions for yes when stroke = yes and stroke = no
summary(probs_gbm[test$stroke == "yes"])
summary(probs_gbm[test$stroke == "no"])

# Make predictions with a threshold of 0.63 for yes
pred_gbm_63 <- as.factor(ifelse(probs_gbm > 0.63, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_gbm_63, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("gbm_63", acc, sens, spec))

# Make predictions with a threshold of 0.62 for yes
pred_gbm_62 <- as.factor(ifelse(probs_gbm > 0.62, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_gbm_62, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("gbm_62", acc, sens, spec))

#=========================#
# nnet model with weights #
#=========================#

# Adjust train control function
ctrl$number <- 5
ctrl$repeats <- 3

# Train weighted nnet model
# Note this will take a minute
set.seed(1001, sample.kind = "Rounding")
train_nnet <- train(stroke ~ .,
                     method = "nnet",
                     data = train,
                     weights = model_weights,
                     metric = "ROC",
                     trControl = ctrl)

# Save model
save(train_nnet, file = "rdas/train_nnet.rda")

# Examine variable importance
varImp(train_nnet)

# Make predictions on test set with weighted nnet model
pred_nnet <- predict(train_nnet, test)

# Examine confusion matrix of results
cm <- confusionMatrix(pred_nnet, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("nnet", acc, sens, spec))

# Extract ROC data from model
test_roc <- roc(test$stroke,
                predict(train_nnet, test, type = "prob")[, "yes"])

# Calculate AUC
auc(test_roc)

# Add AUC to auc_results
auc_results <- rbind(auc_results, c("nnet", auc(test_roc)))

# Pull true positive and false positive rates from test_roc
df_roc_nnet <- data.frame(tpr = rev(test_roc$sensitivities),
                          fpr = rev(1 - test_roc$specificities),
                          model = "nnet")

# Add ROC data from nnet model to df_roc
df_roc <- rbind(df_roc, df_roc_nnet)

# Compare ROC curves from different models
df_roc %>%
  ggplot(aes(fpr, tpr, color = model)) +
  geom_line(size = 1,
            alpha = 0.7) +
  geom_abline(color = "gray") +
  geom_abline(intercept = 1, slope = -1, color = "gray") +
  xlab("false positive rate") +
  ylab("true positive rate") +
  ggtitle("ROC curve by model")

# Save plot
ggsave("figs/roc_models_5.png", dpi = 95)

# Predict probabilities on test set with weighted nnet model
probs_nnet <- predict(train_nnet, test, type = "prob")$yes

# Examine probability distributions for yes when stroke = yes and stroke = no
summary(probs_nnet[test$stroke == "yes"])
summary(probs_nnet[test$stroke == "no"])

# Make predictions with a threshold of 0.59 for yes
pred_nnet_59 <- as.factor(ifelse(probs_nnet > 0.59, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_nnet_59, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("nnet_59", acc, sens, spec))

# Make predictions with a threshold of 0.54 for yes
pred_nnet_54 <- as.factor(ifelse(probs_nnet > 0.54, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_nnet_54, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("nnet_54", acc, sens, spec))

#=============================#
# multinom model with weights #
#=============================#

# Adjust train control function
ctrl$number <- 10
ctrl$repeats <- 5

# Train weighted multinom model
# Note this will take a minute
set.seed(1033, sample.kind = "Rounding")
train_multinom <- train(stroke ~ .,
                    method = "multinom",
                    data = train,
                    weights = model_weights,
                    metric = "ROC",
                    trControl = ctrl)

# Save model
save(train_multinom, file = "rdas/train_multinom.rda")

# Examine variable importance
varImp(train_multinom)

# Make predictions on test set with weighted multinom model
pred_multinom <- predict(train_multinom, test)

# Examine confusion matrix of results
cm <- confusionMatrix(pred_multinom, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("multinom", acc, sens, spec))

# Extract ROC data from model
test_roc <- roc(test$stroke,
                predict(train_multinom, test, type = "prob")[, "yes"])

# Calculate AUC
auc(test_roc)

# Add AUC to auc_results
auc_results <- rbind(auc_results, c("multinom", auc(test_roc)))

# Pull true positive and false positive rates from test_roc
df_roc_multinom <- data.frame(tpr = rev(test_roc$sensitivities),
                              fpr = rev(1 - test_roc$specificities),
                              model = "multinom")

# Add ROC data from multinom model to df_roc
df_roc <- rbind(df_roc, df_roc_multinom)

# Compare ROC curves from different models
df_roc %>%
  ggplot(aes(fpr, tpr, color = model)) +
  geom_line(size = 1,
            alpha = 0.7) +
  geom_abline(color = "gray") +
  geom_abline(intercept = 1,
              slope = -1,
              color = "gray") +
  xlab("false positive rate") +
  ylab("true positive rate") +
  ggtitle("ROC curve by model")

# Save plot
ggsave("figs/roc_models_6.png", dpi = 95)

# Predict probabilities on test set with weighted multinom model
probs_multinom <- predict(train_multinom, test, type = "prob")$yes

# Examine probability distributions for yes when stroke = yes and stroke = no
summary(probs_multinom[test$stroke == "yes"])
summary(probs_multinom[test$stroke == "no"])

# Make predictions with a threshold of 0.56 for yes
pred_multinom_56 <- as.factor(ifelse(probs_multinom > 0.56, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_multinom_56, test$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("multinom_56", acc, sens, spec))

#=============================#
# Build risk assessment model #
#=============================#

# Calculate average stroke rate for training set
mu <- mean(train_strat$stroke == "yes")
mu

#---------------------------------------------------------------#
# Calculate risk factors for each variable and build risk model #
#---------------------------------------------------------------#

# Calculate gender risk factor
r_g <- train_strat %>%
  group_by(gender) %>%
  summarize(r_g = mean(stroke == "yes") - mu)

# Calculate age risk factor
r_a <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  group_by(age) %>%
  summarize(r_a = mean(mean(stroke == "yes") - r_g) - mu)

# Calculate hypertension risk factor
r_ht <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  group_by(hypertension) %>%
  summarize(r_ht = mean(mean(stroke == "yes") - r_g - r_a) - mu)

# Calculate heart disease risk factor
r_hd <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  group_by(heart_disease) %>%
  summarize(r_hd = mean(mean(stroke == "yes") - r_g - r_a - r_ht) - mu)

# Calculate marriage status risk factor
r_m <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  left_join(r_hd, by = "heart_disease") %>%
  group_by(ever_married) %>%
  summarize(r_m = mean(mean(stroke == "yes") - r_g - r_a - r_ht - r_hd) - mu)

# Calculate work type risk factor
r_w <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  left_join(r_hd, by = "heart_disease") %>%
  left_join(r_m, by = "ever_married") %>%
  group_by(work_type) %>%
  summarize(r_w = mean(mean(stroke == "yes") -
                       r_g - r_a - r_ht - r_hd - r_m) - mu)

# Calculate residence type risk factor
r_r <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  left_join(r_hd, by = "heart_disease") %>%
  left_join(r_m, by = "ever_married") %>%
  left_join(r_w, by = "work_type") %>%
  group_by(Residence_type) %>%
  summarize(r_r = mean(mean(stroke == "yes") -
                       r_g - r_a - r_ht - r_hd - r_m - r_w) - mu)

# Calculate average glucose level risk factor
r_ag <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  left_join(r_hd, by = "heart_disease") %>%
  left_join(r_m, by = "ever_married") %>%
  left_join(r_w, by = "work_type") %>%
  left_join(r_r, by = "Residence_type") %>%
  group_by(avg_glucose_level) %>%
  summarize(r_ag = mean(mean(stroke == "yes") -
                        r_g - r_a - r_ht - r_hd - r_m - r_w - r_r) - mu)

# Calculate bmi risk factor
r_b <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  left_join(r_hd, by = "heart_disease") %>%
  left_join(r_m, by = "ever_married") %>%
  left_join(r_w, by = "work_type") %>%
  left_join(r_r, by = "Residence_type") %>%
  left_join(r_ag, by = "avg_glucose_level") %>%
  group_by(bmi) %>%
  summarize(r_b = mean(mean(stroke == "yes") -
                        r_g - r_a - r_ht - r_hd - r_m - r_w - r_r - r_ag) - mu)

# Calculate smoking status risk factor
r_s <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  left_join(r_hd, by = "heart_disease") %>%
  left_join(r_m, by = "ever_married") %>%
  left_join(r_w, by = "work_type") %>%
  left_join(r_r, by = "Residence_type") %>%
  left_join(r_ag, by = "avg_glucose_level") %>%
  left_join(r_b, by = "bmi") %>%
  group_by(smoking_status) %>%
  summarize(r_s = mean(mean(stroke == "yes") -
                       r_g - r_a - r_ht - r_hd - r_m -
                       r_w - r_r - r_ag - r_b) - mu)

# Calculate risk factor assessment for training set
risk <- train_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  left_join(r_hd, by = "heart_disease") %>%
  left_join(r_m, by = "ever_married") %>%
  left_join(r_w, by = "work_type") %>%
  left_join(r_r, by = "Residence_type") %>%
  left_join(r_ag, by = "avg_glucose_level") %>%
  left_join(r_b, by = "bmi") %>%
  left_join(r_s, by = "smoking_status") %>%
  mutate(risk = mu + r_g + r_a + r_ht + r_hd + r_m
         + r_w + r_r + r_ag + r_b + r_s) %>%
  .$risk

# Add risk column to train_strat
train_strat <- train_strat %>%
  mutate(risk = risk)

# Examine distribution of risk values by stroke occurrence for train set
summary(risk[train_strat$stroke == "yes"])
summary(risk[train_strat$stroke == "no"])

# Examine lowest risk values > 0.071 when stroke = "yes"
risk[risk > 0.071 & train$stroke == "yes"] %>%
  sort() %>%
  head()

# Make stroke prediction based on risk cutoff of 0.072
train_strat <- train_strat %>%
  mutate(pred = as.factor(ifelse(risk > 0.072, "yes", "no")))

# Examine confusion matrix of results
confusionMatrix(train_strat$pred, train_strat$stroke, positive = "yes")

# Calculate risk assessments for test set
risk_test <- test_strat %>%
  left_join(r_g, by = "gender") %>%
  left_join(r_a, by = "age") %>%
  left_join(r_ht, by = "hypertension") %>%
  left_join(r_hd, by = "heart_disease") %>%
  left_join(r_m, by = "ever_married") %>%
  left_join(r_w, by = "work_type") %>%
  left_join(r_r, by = "Residence_type") %>%
  left_join(r_ag, by = "avg_glucose_level") %>%
  left_join(r_b, by = "bmi") %>%
  left_join(r_s, by = "smoking_status") %>%
  mutate(risk = mu + r_g + r_a + r_ht + r_hd + r_m
         + r_w + r_r + r_ag + r_b + r_s) %>%
  .$risk

# Make predictions on test set with cutoff of 0.072
pred_risk <- factor(ifelse(risk_test > 0.072, "yes", "no"))

# Examine confusion matrix of results
cm <- confusionMatrix(pred_risk, test_strat$stroke, positive = "yes")
cm

# Add results to accuracy_results
acc <- unname(cm$byClass[[11]])
sens <- unname(cm$byClass[[1]])
spec <- unname(cm$byClass[[2]])
accuracy_results <- rbind(accuracy_results, c("risk_072", acc, sens, spec))

# Add pred_risk column to test_strat
test_strat <- test_strat %>%
  mutate(risk = risk_test)

# Calculate max and min values of risk_test for calculating ROC curve data
max <- max(risk_test)
min <- min(risk_test)

# Round max and min to 4 digits, add 0.0001 to max, subtract 0.0001 from min
# This ensures all points are captured
max <- round(max, 4) + 0.0001
min <- round(min, 4) - 0.0001

# Create data for ROC curve
cutoffs <- seq(max, min, -0.0001)
tprs <- sapply(cutoffs, function(c) {
  preds <- factor(ifelse(test_strat$risk < c, "no", "yes"),
                  levels = c("no", "yes"))
  confusionMatrix(preds, test_strat$stroke, positive = "yes")$byClass[[1]]
})
fprs <- sapply(cutoffs, function(c) {
  preds <- factor(ifelse(test_strat$risk < c, "no", "yes"),
                  levels = c("no", "yes"))
  1 - confusionMatrix(preds, test_strat$stroke, positive = "yes")$byClass[[2]]
})
df_roc_risk <- data.frame(tpr = tprs,
                          fpr = fprs,
                          model = "risk")

# Add ROC data from risk model to df_roc
df_roc <- rbind(df_roc, df_roc_risk)

# Compare ROC curves from different models
df_roc %>%
  ggplot(aes(fpr, tpr, color = model)) +
  geom_line(size = 1,
            alpha = 0.7) +
  geom_abline(color = "gray") +
  geom_abline(intercept = 1,
              slope = -1,
              color = "gray") +
  xlab("false positive rate") +
  ylab("true positive rate") +
  ggtitle("ROC curve by model")

# Save plot
ggsave("figs/roc_models_7.png", dpi = 95)

# Calculate AUC
auc = 0
for (i in 2:length(tprs)) {
  width = fprs[i] - fprs[i - 1]
  height = (tprs[i] + tprs[i - 1]) / 2
  area = width * height
  auc = auc + area
}
auc

# Add AUC to auc_results
auc_results <- rbind(auc_results, c("risk", auc))

######################
# Examine best model #
######################

# Reorder auc_results from lowest to highest
ind <- order(auc_results$AUC)
auc_results <- as_tibble(auc_results[ind,])
auc_results

# Reorder accuracy_results according to balanced accuracy
ind <- order(accuracy_results$balanced_accuracy)
accuracy_results <- as_tibble(accuracy_results[ind,])
accuracy_results

#================================================================#
# Compare nnet probability distributions for test and train sets #
#================================================================#

# Calculate probabilities for train set
probs_train <- predict(train_nnet, train, type = "prob")$yes

# COmpare probability distributions for stroke = "yes"
summary(probs_nnet[test$stroke == "yes"])
summary(probs_train[train$stroke == "yes"])

# Compare probability distributions for stroke = "no"
summary(probs_nnet[test$stroke == "no"])
summary(probs_train[train$stroke == "no"])

#=============================================#
# Simulate validation test with nnet_54 model #
#=============================================#

# Set up 5 iterations of validation test
B <- 5
N <- 10000
set.seed(777, sample.kind = "Rounding")
results <- replicate(B, {
  # Create simulated validation set from data
  ind <- sample(1:5110, N, replace = TRUE)
  val <- dat[ind,]
  
  # Make predictions on validation set with nnet_54 model
  probs <- predict(train_nnet, val, type = "prob")$yes
  preds <- as.factor(ifelse(probs > 0.54, "yes", "no"))
  
  # Extract statistics from confusion matrix
  sens <- confusionMatrix(preds, val$stroke, positive = "yes")$byClass[[1]]
  spec <- confusionMatrix(preds, val$stroke, positive = "yes")$byClass[[2]]
  acc <- confusionMatrix(preds, val$stroke, positive = "yes")$byClass[[11]]
  
  # Return vector with results
  c(sens, spec, acc)
})

# Calculate averages for each row of results
sens <- mean(results[1,])
spec <- mean(results[2,])
acc <- mean(results[3,])

# Add results of final test to accuracy_results
accuracy_results <- rbind(accuracy_results,
                          c("final test nnet_54", acc, sens, spec))

# Display results
accuracy_results

# Convert accuracy results to numeric and round to 4 digits
accuracy_results <- accuracy_results %>%
  mutate_all(parse_guess)
accuracy_results[,2:4] <- round(accuracy_results[,2:4], 4)

# Convert AUC results to numeric and round to 4 digits
auc_results <- auc_results %>%
  mutate_all(parse_guess)
auc_results[,2] <- round(auc_results[,2], 4)

# Save final results
save(accuracy_results, file = "rdas/accuracy_results.rda")
save(auc_results, file = "rdas/auc_results.rda")

#=====================================================#
# Create stroke risk level thresholds for final model #
#=====================================================#

# Predict probabilities for stroke = yes on full data set
probs <- predict(train_nnet, dat, type = "prob")$yes

# Round max value of probs when stroke = "no" down to 2 digits
# Subtract 0.01 to remove last point from the following plot
max <- floor(100 * max(probs[dat$stroke == "no"])) / 100 - 0.01

# Determine proportion ratios for range of thresholds
# Proportion of people with strokes over threshold divided by 
# Proportion of people without strokes over threshold
values <- seq(0, max, 0.01)
prop_ratios <- sapply(values, function(v) {
  prop_yes <- mean(probs[dat$stroke == "yes"] > v)
  prop_no <- mean(probs[dat$stroke == "no"] > v)
  prop_yes / prop_no
})

# Plot proportion ratios vs thresholds
df <- data.frame(threshold = values,
                 ratio = prop_ratios)
df %>%
  ggplot(aes(threshold, ratio)) +
  geom_point(color = "black",
             fill = "indianred4",
             size = 2,
             shape = 21,
             alpha = 0.5) +
  geom_smooth(color = "indianred4",
              size = 1,
              se = FALSE) +
  xlab(expression(paste("probability threshold, ", p[t]))) +
  ylab(expression(paste("Pr(p > ", p[t], " | stroke=yes) / Pr(p > ",
                        p[t], " | stroke=no)"))) +
  scale_y_continuous(breaks = seq(1, 7, 2)) +
  ggtitle("Proportion ratio versus probability threshold")

# Save plot
ggsave("figs/prop_ratio_vs_threshold.png", dpi = 95)

# Find the ratio for threshold 0.54
ratio_54 <- df %>%
  filter(threshold == 0.54) %>%
  .$ratio %>%
  round(2)
ratio_54

# Set ratio markers for risk levels
markers <- c(1.1, 2, ratio_54, 5)

# Find probability threshold values that correspond to ratio markers
p_t <- sapply(markers, function(m) {
  ind <- first(which(df$ratio > m))
  df[ind, 1]
})

# Assign stroke risk level to dat as ordered factor
levels = c("very low", "low", "moderate", "high", "very high")
risk_level <- data.frame(prob = probs) %>%
  mutate(risk_level = case_when(prob <= p_t[1] ~ levels[1],
                                prob > p_t[1] & prob <= p_t[2] ~ levels[2],
                                prob > p_t[2] & prob <= p_t[3] ~ levels[3],
                                prob > p_t[3] & prob <= p_t[4] ~ levels[4],
                                prob > p_t[4] ~ levels[5])) %>%
  mutate(risk_level = factor(risk_level, levels = levels, ordered = TRUE))

# Load dat_clean for ease of exploration
load("rdas/dat_clean.rda")

# Set stroke levels to no and yes
levels(dat$stroke) <- c("no", "yes")

# Join dat with risk level
dat <- cbind(dat, risk_level)

# Add prediction column to dat
dat <- dat %>%
  mutate(pred = as.factor(ifelse(prob > 0.54, "yes", "no")))

# Save final risk assessment
save(dat, file = "rdas/final_risk_assessment.rda")

#============================#
# Explore nnet_54 final data #
#============================#

# Table risk levels of people who have had strokes
table(dat$risk_level[dat$stroke == "yes"])

# Inspect case where stroke = yes and risk level = very low
dat %>% filter(stroke == "yes" & risk_level == "very low")

# Table risk levels of people who have not had strokes
table(dat$risk_level[dat$stroke == "no"])

# Calculate proportion of stroke = no with risk = high or very high
dat %>%
  filter(stroke == "no") %>%
  summarize(mean(risk_level %in% c("high", "very high")))

# Create data frames for true and false positives and negatives
fns <- dat %>%
  filter(stroke == "yes" & prob <= 0.54) %>%
  mutate(result = "false neg")
tns <- dat %>%
  filter(stroke == "no" & prob <= 0.54) %>%
  mutate(result = "true neg")
fps <- dat %>%
  filter(stroke == "no" & prob > 0.54) %>%
  mutate(result = "false pos")
tps <- dat %>%
  filter(stroke == "yes" & prob > 0.54) %>%
  mutate(result = "true pos")

# Add result = "all data" column to dat
dat <- dat %>%
  mutate(result = "all data")

# Join dat with fns, tns, fps, and tps
dat <- dat %>%
  rbind(fns, tns, fps, tps)

# Convert results column to factor
levels <- c("true pos", "false pos", "all data", "true neg", "false neg")
dat$result <- factor(dat$result, levels = levels)

# Plot gender distributions by result
dat %>%
  group_by(result) %>%
  summarize(n_male = sum(gender == "Male"),
            n_female = sum(gender == "Female"),
            n_other = sum(gender == "Other"),
            male = 100 * n_male / n(),
            female = 100 * n_female / n(),
            other = 100 * n_other / n()) %>%
  select(result, male:other) %>%
  gather(gender, percent, -result) %>%
  group_by(result) %>%
  ggplot(aes(result, percent, fill = gender)) +
  geom_bar(color = "black",
           stat = "identity",
           position = "dodge") +
  scale_fill_manual(values = c("orchid3", "skyblue3", "seagreen3")) +
  xlab("prediction result") +
  scale_y_continuous(labels = paste0(seq(0, 60, 20), "%")) +
  ggtitle("Gender distribution by prediction result")
  
# Save plot
ggsave("figs/gender_distribution_by_result.png", dpi = 95)

# Plot age distributions by result
dat %>%
  mutate(result = reorder(result, -as.numeric(result))) %>%
  ggplot(aes(result, age)) +
  geom_boxplot(color = "black",
               fill = "slateblue3") +
  xlab("prediction result") +
  coord_flip() +
  ggtitle("Age distribution by prediction result")

# Save plot
ggsave("figs/age_distribution_by_result.png", dpi = 95)

# Plot hypertension distribution by result
dat %>%
  group_by(result) %>%
  summarize(n_yes = sum(hypertension == 1),
            n_no = sum(hypertension == 0),
            yes = 100 * n_yes / n(),
            no = 100 * n_no / n()) %>%
  select(result, yes, no) %>%
  gather(hypertension, percent, -result) %>%
  group_by(result) %>%
  ggplot(aes(result, percent, fill = hypertension)) +
  geom_bar(color = "black",
           stat = "identity",
           position = "dodge") +
  scale_fill_manual(values = c("royalblue3", "brown2")) +
  xlab("prediction result") +
  scale_y_continuous(labels = paste0(seq(0, 100, 25), "%")) +
  ggtitle("Hypertension distribution by prediction result")

# Save plot
ggsave("figs/hypertension_distribution_by_result.png", dpi = 95)

# Plot heart disease distribution by result
dat %>%
  group_by(result) %>%
  summarize(n_yes = sum(heart_disease == 1),
            n_no = sum(heart_disease == 0),
            yes = 100 * n_yes / n(),
            no = 100 * n_no / n()) %>%
  select(result, yes, no) %>%
  gather(heart_disease, percent, -result) %>%
  group_by(result) %>%
  ggplot(aes(result, percent, fill = heart_disease)) +
  geom_bar(color = "black",
           stat = "identity",
           position = "dodge") +
  scale_fill_manual(name = "heart disease",
                    values = c("royalblue3", "brown2")) +
  xlab("prediction result") +
  scale_y_continuous(labels = paste0(seq(0, 100, 25), "%")) +
  ggtitle("Heart disease distribution by prediction result")

# Save plot
ggsave("figs/heart_disease_distribution_by_result.png", dpi = 95)

# Plot marriage status distribution by result
dat %>%
  group_by(result) %>%
  summarize(n_yes = sum(ever_married == "Yes"),
            n_no = sum(ever_married == "No"),
            yes = 100 * n_yes / n(),
            no = 100 * n_no / n()) %>%
  select(result, yes, no) %>%
  gather(ever_married, percent, -result) %>%
  group_by(result) %>%
  ggplot(aes(result, percent, fill = ever_married)) +
  geom_bar(color = "black",
           stat = "identity",
           position = "dodge") +
  scale_fill_manual(name = "ever married",
                    values = c("seagreen3", "orchid3")) +
  xlab("prediction result") +
  scale_y_continuous(labels = paste0(seq(0, 100, 25), "%")) +
  ggtitle("Marriage status distribution by prediction result")

# Save plot
ggsave("figs/marriage_status_distribution_by_result.png", dpi = 95)

# Plot work type distribution by result
dat %>%
  group_by(result) %>%
  summarize(n_private = sum(work_type == "Private"),
            n_self = sum(work_type == "Self-employed"),
            n_child = sum(work_type == "children"),
            n_govt = sum(work_type == "Govt_job"),
            n_never = sum(work_type == "Never_worked"),
            a_private = 100 * n_private / n(),
            b_self = 100 * n_self / n(),
            c_child = 100 * n_child / n(),
            d_govt = 100 * n_govt / n(),
            e_never = 100 * n_never / n()) %>%
  select(result, a_private:e_never) %>%
  gather(work_type, percent, -result) %>%
  group_by(result) %>%
  ggplot(aes(result, percent, fill = work_type)) +
  geom_bar(color = "black",
           stat = "identity",
           position = "dodge") +
  scale_fill_manual(name = "work type",
                    labels = c("private", "self-employed", "children",
                               "govt job", "never worked"),
                    values = c("firebrick2", "chocolate2", "seagreen3",
                               "slateblue3", "orchid3")) +
  xlab("prediction result") +
  scale_y_continuous(breaks = seq(0, 60, 20),
                     labels = paste0(seq(0, 60, 20), "%")) +
  ggtitle("Work type distribution by prediction result")

# Save plot
ggsave("figs/work_type_distribution_by_result.png", dpi = 95)

# Plot residence type distribution by result
dat %>%
  group_by(result) %>%
  summarize(n_urban = sum(Residence_type == "Urban"),
            n_rural = sum(Residence_type == "Rural"),
            urban = 100 * n_urban / n(),
            rural = 100 * n_rural / n()) %>%
  select(result, urban, rural) %>%
  gather(Residence_type, percent, -result) %>%
  group_by(result) %>%
  ggplot(aes(result, percent, fill = Residence_type)) +
  geom_bar(color = "black",
           stat = "identity",
           position = "dodge") +
  scale_fill_manual(name = "residence type",
                    values = c("tan4", "steelblue3")) +
  xlab("prediction result") +
  scale_y_continuous(breaks = c(0, 20, 40),
                     labels = paste0(c(0, 20, 40), "%")) +
  ggtitle("Residence type distribution by prediction result")

# Save plot
ggsave("figs/residence_type_distribution_by_result.png", dpi = 95)

# Plot average glucose level distribution by result
dat %>%
  mutate(result = reorder(result, -as.numeric(result))) %>%
  ggplot(aes(result, avg_glucose_level)) +
  geom_boxplot(color = "black",
               fill = "orchid3") +
  xlab("prediction result") +
  ylab("average glucose level") +
  coord_flip() +
  ggtitle("Average glucose level distribution by prediction result")

# Save plot
ggsave("figs/glucose_distribution_by_result.png", dpi = 95)

# Plot bmi distribution by result
dat %>%
  mutate(result = reorder(result, -as.numeric(result))) %>%
  ggplot(aes(result, bmi)) +
  geom_boxplot(color = "black",
               fill = "seagreen3") +
  xlab("prediction result") +
  ylab("body mass index") +
  coord_flip() +
  ggtitle("Body mass index distribution by prediction result")

# Save plot
ggsave("figs/bmi_distribution_by_result.png", dpi = 95)

# Plot bmi distribution by result cropped
dat %>%
  mutate(result = reorder(result, -as.numeric(result))) %>%
  ggplot(aes(result, bmi)) +
  geom_boxplot(color = "black",
               fill = "seagreen3") +
  xlab("prediction result") +
  ylim(c(10,50)) +
  ylab("body mass index") +
  coord_flip() +
  ggtitle("Body mass index distribution by prediction result")

# Save plot
ggsave("figs/bmi_distribution_by_result_cropped.png", dpi = 95)

# Plot smoking status distribution by result
dat %>%
  group_by(result) %>%
  summarize(n_never = sum(smoking_status == "never smoked"),
            n_unknown = sum(smoking_status == "Unknown"),
            n_formerly = sum(smoking_status == "formerly smoked"),
            n_smokes = sum(smoking_status == "smokes"),
            a_never = 100 * n_never / n(),
            b_unknown = 100 * n_unknown / n(),
            c_formerly = 100 * n_formerly / n(),
            d_smokes = 100 * n_smokes / n()) %>%
  select(result, a_never:d_smokes) %>%
  gather(smoking_status, percent, -result) %>%
  group_by(result) %>%
  ggplot(aes(result, percent, fill = smoking_status)) +
  geom_bar(color = "black",
           stat = "identity",
           position = "dodge") +
  scale_fill_manual(name = "smoking status",
                    labels = c("never smoked", "unknown", "formerly smoked",
                               "smokes"),
                    values = c("seagreen3", "slateblue3",
                               "orchid3", "firebrick3")) +
  xlab("prediction result") +
  scale_y_continuous(labels = paste0(seq(0, 40, 10), "%")) +
  ggtitle("Smoking status distribution by prediction result")

# Save plot
ggsave("figs/smoking_status_distribution_by_result.png", dpi = 95)
