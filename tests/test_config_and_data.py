from project2_app.config import load_config
from project2_app.data import load_dataset, split_dataset
from project2_app.fingerprint import experiment_fingerprint


def test_config_and_dataset_contract() -> None:
    settings = load_config()
    dataset = load_dataset()
    split = split_dataset(dataset, settings)

    assert len(settings.team_members) == 4
    assert len(dataset.X) == 569
    assert dataset.X.shape[1] == 30
    assert set(dataset.y.unique()) == {0, 1}
    assert dataset.y.name == "malignant"
    assert len(split.X_train) + len(split.X_test) == 569
    assert len(experiment_fingerprint(settings)) == 16


def test_stratified_split_is_reproducible() -> None:
    settings = load_config()
    dataset = load_dataset()
    split_a = split_dataset(dataset, settings)
    split_b = split_dataset(dataset, settings)

    assert split_a.X_test.index.tolist() == split_b.X_test.index.tolist()
    assert split_a.y_test.tolist() == split_b.y_test.tolist()
