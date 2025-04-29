TRAIN_RATIO = 0.75
VAL_RATIO = 0.2
TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

BASIS_IDENTIFIERS = 100000
CLASS_IDENTIFIERS = {
    'car': BASIS_IDENTIFIERS*10,
    'bus': BASIS_IDENTIFIERS*20,
    'truck': BASIS_IDENTIFIERS*30,
    'rider': BASIS_IDENTIFIERS*40,
    'bicycle': BASIS_IDENTIFIERS*50,
    'motorcycle': BASIS_IDENTIFIERS*60,
    'pedestrian': BASIS_IDENTIFIERS*70,
}

YOLO_LABEL_IDS = {
    10: 0,
    20: 1,
    30: 2,
    40: 3,
    50: 4,
    60: 5,
    70: 3,
}

# for create data.yaml
CLASS_ID_NAMES = {
    0: 'car', 
    1: 'bus', 
    2: 'truck', 
    3: 'person',  # person = rider + pedestrian
    4: 'bicycle', 
    5: 'motorcycle',
} 


