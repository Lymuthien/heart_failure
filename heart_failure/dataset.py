from sklearn.model_selection import train_test_split
from heart_failure.config.features import RANDOM_STATE, VAL_SIZE, TEST_SIZE

def split_data(*arrays, stratify_func: callable):
    stratify = stratify_func(*arrays)
    train_test = train_test_split(
        *arrays,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    train_arrays = train_test[::2]
    test_arrays = train_test[1::2]

    val_stratify = stratify_func(*train_arrays)
    train_val = train_test_split(
        *train_arrays,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=val_stratify,
    )
    train_arrays = train_val[::2]
    val_arrays = train_val[1::2]

    return *train_arrays, *val_arrays, *test_arrays
