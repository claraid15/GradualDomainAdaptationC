import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

def rand_seed(seed):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def self_train_once(student, teacher, unsup_x, confidence_q=0.1, epochs=20):
    # Do one bootstrapping step on unsup_x, where pred_model is used to make predictions,
    # and we use these predictions to update model.
    logits = teacher.predict(np.concatenate([unsup_x])) #logits is a two-dimensional array, each row is the model's predicted probability for each category of the sample
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)
    alpha = np.quantile(confidence, confidence_q)
    indices = np.argwhere(confidence >= alpha)[:, 0]
    preds = np.argmax(logits, axis=1) #For each sample’s predicted probability distribution, select the category with the highest probability as the model’s predicted category, that is, generate a pseudo label for the sample
    student.fit(unsup_x[indices], preds[indices], epochs=epochs, verbose=False) #Train the student model with the selected high confidence samples and their corresponding pseudo labels

def self_train_once_simple(student, teacher, unsup_x, confidence_q=0.1):
    # Perform a self-training step to generate predicted pseudo labels using the teacher model
    logits = teacher.predict_proba(np.concatenate([unsup_x]))  
    confidence = np.amax(logits, axis=1) - np.amin(logits, axis=1)  
    alpha = np.quantile(confidence, confidence_q) 
    indices = np.argwhere(confidence >= alpha)[:, 0]  
    preds = np.argmax(logits, axis=1) 
    student.fit(unsup_x[indices], preds[indices])  


#Take the predicted probability distribution probs (soft labels) of all unlabeled data as the target and train the student model
def soft_self_train_once(student, teacher, unsup_x, epochs=20):
    probs = teacher.predict(np.concatenate([unsup_x]))
    student.fit(unsup_x, probs, epochs=epochs, verbose=False)

def soft_self_train_once_simple(student, teacher, unsup_x):
    probs = teacher.predict_proba(np.concatenate([unsup_x]))
    # For simple models, do not use soft labels directly for training
    # Here we assume that you use hard labels for training directly. If you use a model that supports soft labels, you can keep the soft labels
    preds = np.argmax(probs, axis=1)  
    student.fit(unsup_x, preds)  



def self_train(student_func, teacher, unsup_x, confidence_q=0.1, epochs=20, repeats=1,
               target_x=None, target_y=None, soft=False):
    accuracies = []
    for i in range(repeats):
        student = student_func(teacher)
        if soft:
            soft_self_train_once(student, teacher, unsup_x, epochs)
        else:
            self_train_once(student, teacher, unsup_x, confidence_q, epochs)
        if target_x is not None and target_y is not None:
            _, accuracy = student.evaluate(target_x, target_y, verbose=True)
            accuracies.append(accuracy)
        teacher = student
    return accuracies, student

def self_train_simple(student_func, teacher, unsup_x, confidence_q=0.1, repeats=1,
               target_x=None, target_y=None, soft=False):
    accuracies = []
    for i in range(repeats):
        student = student_func(teacher)
        if soft:
            soft_self_train_once_simple(student, teacher, unsup_x)
        else:
            self_train_once_simple(student, teacher, unsup_x, confidence_q)
        if target_x is not None and target_y is not None:
            accuracy = student.score(target_x, target_y)  
            accuracies.append(accuracy)
        teacher = student
    return accuracies, student



def gradual_self_train(student_func, teacher, unsup_x, debug_y, interval, confidence_q=0.1,
                       epochs=20, soft=False):
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []
    for i in range(upper_idx):
        student = student_func(teacher)
        cur_xs = unsup_x[interval*i:interval*(i+1)]
        cur_ys = debug_y[interval*i:interval*(i+1)] #debug_y为真实标签 
        # _, student = self_train(
        #     student_func, teacher, unsup_x, confidence_q, epochs, repeats=2)
        if soft:
            soft_self_train_once(student, teacher, cur_xs, epochs)
        else:
            self_train_once(student, teacher, cur_xs, confidence_q, epochs)
        _, accuracy = student.evaluate(cur_xs, cur_ys)
        accuracies.append(accuracy)
        teacher = student
    return accuracies, student

def gradual_self_train_simple(student_func, teacher, unsup_x, inter_y, interval, confidence_q=0.1,
                       soft=False):
    upper_idx = int(unsup_x.shape[0] / interval)
    accuracies = []
    for i in range(upper_idx):
        student = student_func(teacher)
        cur_xs = unsup_x[interval * i:interval * (i + 1)]
        cur_ys = inter_y[interval * i:interval * (i + 1)]
        if soft:
            soft_self_train_once_simple(student, teacher, cur_xs)
        else:
            self_train_once_simple(student, teacher, cur_xs, confidence_q)
        accuracy = student.score(cur_xs, cur_ys)  
        accuracies.append(accuracy)
        teacher = student
    return accuracies, student

def split_data(xs, ys, splits):
    return np.split(xs, splits), np.split(ys, splits)


def train_to_acc(model, acc, train_x, train_y, val_x, val_y):
    # Modify steps per epoch to be around dataset size / 10
    # Keep training until accuracy 
    batch_size = 32
    data_size = train_x.shape[0]
    steps_per_epoch = int(data_size / 50.0 / batch_size)
    logger.info("train_xs size is %s", str(train_x.shape))
    while True:
        model.fit(train_x, train_y, batch_size=batch_size, steps_per_epoch=steps_per_epoch, verbose=False)
        val_accuracy = model.evaluate(val_x, val_y, verbose=False)[1]
        logger.info("validation accuracy is %f", val_accuracy)
        if val_accuracy >= acc:
            break
    return model


def save_model(model, filename):
    model.save(filename)


def load_model(filename):
    model = load_model(filename)


def rolling_average(sequence, r):
    N = sequence.shape[0]
    assert r < N
    assert r > 1
    rolling_sums = []
    cur_sum = sum(sequence[:r])
    rolling_sums.append(cur_sum)
    for i in range(r, N):
        cur_sum = cur_sum + sequence[i] - sequence[i-r]
        rolling_sums.append(cur_sum)
    return np.array(rolling_sums) * 1.0 / r

