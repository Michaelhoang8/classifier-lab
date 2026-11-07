import math
import random
import statistics
from .data_apps import parse
from .validation import ValidationError, integer

def number(value, name, low, high):
    if isinstance(value, bool):
        raise ValidationError(f'{name} must be numeric')
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValidationError(f'{name} must be numeric') from None
    if not math.isfinite(result) or not low <= result <= high:
        raise ValidationError(f'{name} must be finite and between {low} and {high}')
    return result

def auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(scores, labels))
    rank_sum, start = (0, 0)
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        rank_sum += sum((y for _, y in pairs[start:end])) * (start + 1 + end) / 2
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

def threshold_metrics(labels, scores, threshold=0.5):
    tp = fp = tn = fn = 0
    for y, p in zip(labels, scores):
        predicted = p >= threshold
        tp += y == 1 and predicted
        fp += y == 0 and predicted
        tn += y == 0 and (not predicted)
        fn += y == 1 and (not predicted)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'accuracy': (tp + tn) / len(labels), 'precision': precision, 'recall': recall, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None, 'confusion': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}}

def probability_metrics(labels, scores):
    return {'roc_auc': auc(labels, scores), 'brier': statistics.mean(((p - y) ** 2 for y, p in zip(labels, scores))), 'log_loss': -statistics.mean((y * math.log(max(1e-15, p)) + (1 - y) * math.log(max(1e-15, 1 - p)) for y, p in zip(labels, scores)))}

def metrics(labels, scores, threshold=0.5):
    return {**threshold_metrics(labels, scores, threshold), **probability_metrics(labels, scores)}

def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)

def train(data):
    headers, rows = parse(data)
    target = data.get('target', 'label')
    if target not in headers:
        raise ValidationError('Select an existing target column')
    features = [h for h in headers if h != target]
    if not 1 <= len(features) <= 12 or not 8 <= len(rows) <= 1000:
        raise ValidationError('Training requires 8–1,000 rows and 1–12 numeric feature columns')
    seed = integer(data, 'seed', 0, 1000000) if 'seed' in data else 42
    x, y = ([], [])
    for index, row in enumerate(rows, 2):
        label = number(row[headers.index(target)], f'label at record {index}', 0, 1)
        if label not in (0, 1):
            raise ValidationError('Target labels must be 0 or 1')
        y.append(int(label))
        x.append([number(row[headers.index(h)], f'{h} at record {index}', -1000000000.0, 1000000000.0) for h in features])
    if min(y.count(0), y.count(1)) < 4:
        raise ValidationError('Include at least four examples of each class')
    first_features = {}
    for index, features_row in enumerate(x):
        key = tuple(features_row)
        if key in first_features:
            first = first_features[key]
            detail = 'conflicting target labels' if y[first] != y[index] else 'matching target labels'
            raise ValidationError(f'Duplicate feature rows found at records {first + 2} and {index + 2} after numeric parsing ({detail}). Review these records, then deduplicate or use an appropriate grouped split first')
        first_features[key] = index
    rng = random.Random(seed)
    train_ids, test_ids = ([], [])
    for label in (0, 1):
        ids = [i for i, value in enumerate(y) if value == label]
        rng.shuffle(ids)
        count = max(1, round(len(ids) * 0.25))
        test_ids.extend(ids[:count])
        train_ids.extend(ids[count:])
    means = [statistics.mean((x[i][j] for i in train_ids)) for j in range(len(features))]
    scales = [max(statistics.pstdev((x[i][j] for i in train_ids)) or 1, 1e-12) for j in range(len(features))]
    scaled = [[1, *[(row[j] - means[j]) / scales[j] for j in range(len(features))]] for row in x]
    weights = [0.0] * (len(features) + 1)
    losses = []
    for epoch in range(201):
        predictions = [sigmoid(sum((w * v for w, v in zip(weights, scaled[i])))) for i in train_ids]
        if epoch % 20 == 0:
            losses.append({'epoch': epoch, 'log_loss': metrics([y[i] for i in train_ids], predictions)['log_loss']})
        if epoch == 200:
            break
        gradient = [0.0] * len(weights)
        for i, p in zip(train_ids, predictions):
            for j, value in enumerate(scaled[i]):
                gradient[j] += (p - y[i]) * value
        weights = [w - 0.1 * (gradient[j] / len(train_ids) + (0.01 * w if j else 0)) for j, w in enumerate(weights)]
    predictions = [sigmoid(sum((w * v for w, v in zip(weights, scaled[i])))) for i in test_ids]
    majority = int(sum((y[i] for i in train_ids)) > len(train_ids) / 2)
    return {'train_rows': len(train_ids), 'test_rows': len(test_ids), 'seed': seed, 'metrics': metrics([y[i] for i in test_ids], predictions), 'baseline_accuracy': sum((y[i] == majority for i in test_ids)) / len(test_ids), 'loss_history': losses, 'features': [{'name': name, 'weight': weights[j + 1], 'mean': means[j], 'scale': scales[j]} for j, name in enumerate(features)], 'model': {'algorithm': 'standardized binary logistic regression', 'intercept': weights[0], 'weights': weights[1:], 'features': features, 'means': means, 'scales': scales, 'threshold': 0.5}, 'predictions': [{'record': i + 2, 'actual': y[i], 'score': p} for i, p in zip(test_ids, predictions)], 'train_records': [i + 2 for i in train_ids], 'test_records': [i + 2 for i in test_ids]}
