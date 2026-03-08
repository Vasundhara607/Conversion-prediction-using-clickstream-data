install.packages(c("keras3", "tensorflow", "tfdatasets", "reticulate"))
library(reticulate)
library(tidyverse)
library(keras3)
library(pROC)
library(iml)
library(glmnet)
install.packages("data.table")
library(data.table)
install_miniconda(force = TRUE)
miniconda_path()
conda_create("cis8398", python_version = "3.11")
use_condaenv("cis8398")
py_install(c("pip","setuptools","wheel"), pip = TRUE, upgrade = TRUE)
py_install("numpy", pip = TRUE, upgrade = TRUE)
py_install(c("tensorflow","keras"), pip = TRUE)
install.packages("keras3")
install.packages("tensorflow")
library(ggplot2)
library(yardstick)
set.seed(42)

#Load events_reduced.csv 
EVENTS_PATH <- "C:/Users/vasun/Desktop/CIS8398/events_reduced.csv"
events <- fread(
  EVENTS_PATH,
  select = c("timestamp", "visitorid", "event", "itemid", "transactionid")
)
events[, ts := as.POSIXct(timestamp / 1000, origin = "1970-01-01", tz = "UTC")]
events[, event := tolower(event)]
# sort
setorder(events, visitorid, ts)
cat("Loaded events:", nrow(events), 
    "| Unique visitors:", uniqueN(events$visitorid), "\n")
events[1:5]

# Sessionization (30-min gap)

SESSION_GAP_MINS <- 30

events[, gap_mins := as.numeric(difftime(ts, shift(ts), units = "mins")), by = visitorid]
events[, new_session := fifelse(is.na(gap_mins) | gap_mins > SESSION_GAP_MINS, 1L, 0L), by = visitorid]
events[, session_index := cumsum(new_session), by = visitorid]
events[, session_id := paste0(visitorid, "_", session_index)]

print(events[1:5, .(visitorid, ts, gap_mins, new_session, session_id, event)])

#trim each session

events[, first_txn_ts := suppressWarnings(min(ts[event == "transaction"])), by = session_id]
events_trim <- events[is.na(first_txn_ts) | ts <= first_txn_ts]
cat("Rows before trim:", nrow(events), "\n")
cat("Rows after  trim:", nrow(events_trim), "\n")

#Build session-level dataset (features + label)

sessions <- events_trim[, .(
  session_start  = min(ts),
  session_end    = max(ts),
  duration_secs  = max(1, as.numeric(difftime(max(ts), min(ts), units = "secs"))),
  n_events       = .N,
  n_items        = uniqueN(itemid),
  n_view         = sum(event == "view", na.rm = TRUE),
  n_addtocart    = sum(event == "addtocart", na.rm = TRUE),
  converted      = as.integer(any(event == "transaction"))
), by = .(session_id, visitorid)]

sessions[, duration_min := duration_secs / 60]
sessions[, `:=`(
  events_per_min = n_events / pmax(duration_min, 1),
  view_ratio     = n_view / pmax(n_events, 1),
  cart_ratio     = n_addtocart / pmax(n_events, 1)
)]

setorder(sessions, session_start)

cat("Total sessions:", nrow(sessions), "\n")
print(sessions[, .N, by = converted])
print(sessions[1:5])

# Session-level sampling

NEG_POS_RATIO <- 5   # negatives vs positives=1:5

pos_sids <- sessions[converted == 1, session_id]
neg_sids <- sessions[converted == 0, session_id]
neg_keep <- sample(neg_sids, size = min(length(neg_sids), length(pos_sids) * NEG_POS_RATIO))

keep_sids <- c(pos_sids, neg_keep)

events_keep <- events_trim[session_id %in% keep_sids]

cat("Reduced events (session-safe):", nrow(events_keep), "\n")
cat("Sessions kept:", uniqueN(events_keep$session_id), "\n")

# Rebuild sessions from sampled events
sessions_keep <- events_keep[, .(
  session_start  = min(ts),
  session_end    = max(ts),
  duration_secs  = max(1, as.numeric(difftime(max(ts), min(ts), units = "secs"))),
  n_events       = .N,
  n_items        = uniqueN(itemid),
  n_view         = sum(event == "view", na.rm = TRUE),
  n_addtocart    = sum(event == "addtocart", na.rm = TRUE),
  converted      = as.integer(any(event == "transaction"))
), by = .(session_id, visitorid)]

sessions_keep[, duration_min := duration_secs / 60]
sessions_keep[, `:=`(
  events_per_min = n_events / pmax(duration_min, 1),
  view_ratio     = n_view / pmax(n_events, 1),
  cart_ratio     = n_addtocart / pmax(n_events, 1)
)]
setorder(sessions_keep, session_start)

cat("Sampled sessions:", nrow(sessions_keep), "\n")
print(sessions_keep[, .N, by = converted])

# Class distribution
ggplot(sessions_keep, aes(x = factor(converted))) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Session Conversion Distribution",
    x = "Converted (0 = No, 1 = Yes)",
    y = "Count"
  ) +
  theme_minimal()

# Train/Test split by visitor
vis <- unique(sessions_keep$visitorid)
train_vis <- sample(vis, size = floor(0.8 * length(vis)))

train_df <- sessions_keep[visitorid %in% train_vis]
test_df  <- sessions_keep[!visitorid %in% train_vis]

cat("Train sessions:", nrow(train_df), "| Test sessions:", nrow(test_df), "\n")
cat("Train pos rate:", round(mean(train_df$converted), 4),
    "| Test pos rate:", round(mean(test_df$converted), 4), "\n")

stopifnot(length(intersect(train_df$visitorid, test_df$visitorid)) == 0)
stopifnot(length(intersect(train_df$session_id, test_df$session_id)) == 0)


#Logistic Regression

feature_cols <- c(
  "n_view",
  "duration_min"
)

x_train <- as.matrix(train_df[, ..feature_cols])
y_train <- as.numeric(train_df$converted)

x_test  <- as.matrix(test_df[, ..feature_cols])
y_test  <- as.numeric(test_df$converted)

set.seed(42)
cv_fit <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1,
  type.measure = "auc",
  nfolds = 5
)

best_lambda <- cv_fit$lambda.min
cat("Best lambda:", best_lambda, "\n")

lr_fit <- glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1,
  lambda = best_lambda
)

p_hat <- as.numeric(predict(lr_fit, newx = x_test, type = "response"))

auc_val <- as.numeric(pROC::roc(response = y_test, predictor = p_hat, quiet = TRUE)$auc)
cat("Test AUC (glmnet):", round(auc_val, 4), "\n")

# F1 threshold tuning
eval_tbl <- tibble(truth = factor(y_test, levels = c(0,1)), prob = p_hat)
thresholds <- seq(0.05, 0.95, by = 0.05)

f1_scores <- sapply(thresholds, function(t) {
  pred <- factor(ifelse(p_hat >= t, 1, 0), levels = c(0, 1))
  yardstick::f_meas_vec(truth = eval_tbl$truth, estimate = pred, event_level = "second")
})

best_t <- thresholds[which.max(f1_scores)]
cat("Best F1 threshold:", best_t, "| Best F1:", round(max(f1_scores), 4), "\n")

eval_tbl <- eval_tbl %>% mutate(pred = factor(ifelse(prob >= best_t, 1, 0), levels = c(0,1)))
metrics <- yardstick::metrics(eval_tbl, truth = truth, estimate = pred) %>%
  bind_rows(
    yardstick::precision(eval_tbl, truth, pred),
    yardstick::recall(eval_tbl, truth, pred),
    yardstick::f_meas(eval_tbl, truth, pred)
  )
print(metrics)

#Sequential Encoding for LSTM 

events_seq <- events_keep[event %in% c("view","addtocart")] 
seq_len <- 5
action_map <- c("view" = 1L, "addtocart" = 2L)

get_sequence <- function(sid, dt, mapping, maxlen) {
  evs <- dt[session_id == sid, event]
  nums <- unname(mapping[evs])
  nums[is.na(nums)] <- 0L
  n <- length(nums)
  if (n >= maxlen) return(nums[(n - maxlen + 1):n])
  c(rep(0L, maxlen - n), nums)
}
x_train_seq <- do.call(rbind, lapply(train_df$session_id, get_sequence, events_seq, action_map, seq_len))
x_test_seq  <- do.call(rbind, lapply(test_df$session_id,  get_sequence, events_seq, action_map, seq_len))

y_train_seq <- as.numeric(train_df$converted)
y_test_seq  <- as.numeric(test_df$converted)

stopifnot(nrow(x_train_seq) == length(y_train_seq))
stopifnot(nrow(x_test_seq)  == length(y_test_seq))

cat("Sequence train shape:", paste(dim(x_train_seq), collapse=" x "), "\n")
cat("Sequence test shape :", paste(dim(x_test_seq),  collapse=" x "), "\n")

#LSTM training and evaluation

model_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = 3, output_dim = 8, input_shape = c(seq_len)) %>%
  layer_lstm(units = 32, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model_lstm %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c(
    "accuracy",
    metric_auc(name = "auc"),
    metric_precision(name = "precision"),
    metric_recall(name = "recall")
  )
)

cat("Starting LSTM training...\n")
history <- model_lstm %>% fit(
  x = x_train_seq,
  y = y_train_seq,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1
)

cat("\n--- LSTM Test Metrics ---\n")
results <- model_lstm %>% evaluate(x_test_seq, y_test_seq, verbose = 0)
print(results)

#Compare Accuracy

log_pred <- ifelse(p_hat >= best_t, 1, 0)
log_acc  <- mean(log_pred == y_test)
lstm_prob <- as.numeric(model_lstm %>% predict(x_test_seq))
lstm_pred <- ifelse(lstm_prob >= 0.5, 1, 0) 
lstm_acc  <- mean(lstm_pred == y_test_seq)

# Print accuracies
cat("GLMNET Accuracy (best_t):", round(log_acc, 4), "\n")
cat("LSTM   Accuracy (0.5):  ", round(lstm_acc, 4), "\n")

#Build comparison table
acc_df <- data.table(
  Model = c("Logistic Regression (glmnet)", "LSTM"),
  Accuracy = c(log_acc, lstm_acc)
)

#Plot 
ggplot(acc_df, aes(x = Model, y = Accuracy)) +
  geom_col() +
  geom_text(aes(label = round(Accuracy, 4)), vjust = -0.4, size = 4) +
  ylim(0, 1) +
  labs(
    title = "Accuracy Comparison: Logistic Regression vs LSTM",
    x = "",
    y = "Test Accuracy"
  ) +
  theme_minimal()

#Explainability (XAI)
library(iml)
pred_fun <- function(model, newdata) as.numeric(predict(model, as.matrix(newdata)))
 
# Convert sequence to dataframe for IML
x_test_df <- as.data.frame(x_test_seq)
colnames(x_test_df) <- paste0("Click_", 1:seq_len)
 
predictor <- Predictor$new(model_lstm, data = x_test_df, y = y_test, predict.fun = pred_fun)
 
# Plot Global Importance
plot(FeatureImp$new(predictor, loss = "mae")) + ggtitle("Global Explanation: Importance of Each Click")
 
# Local Explanation: Explain one specific person's purchase
conv_idx <- which(y_test == 1)[1]
shapley <- Shapley$new(predictor, x.interest = x_test_df[conv_idx, ])
plot(shapley) + ggtitle(paste("Local Explanation: Why did user", conv_idx, "convert?"))
