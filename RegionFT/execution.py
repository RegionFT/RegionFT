from RegionFT.partition import PrecomputedPartitioner, load_regions_from_csv
from RegionFT.regionft import RegionFT


def run_regionft(
    cut,
    protected_index,
    runtime,
    label,
    disc_dir,
    test_dir,
    partition_dir,
    show_logging=False,
    config=None,
):
    """Run one RegionFT configuration, optionally reusing a saved partition."""
    regionft_config = dict(config or {})
    partition_file = regionft_config.pop("partition_file", None)
    if partition_file is not None:
        regions = load_regions_from_csv(str(partition_file))
        regionft_config["partitioner"] = PrecomputedPartitioner(regions)

    tester = RegionFT(
        cut,
        [protected_index],
        show_logging=show_logging,
        **regionft_config,
    )
    tester.test(
        runtime=runtime,
        label=label,
        disc_save_to=disc_dir,
        test_save_to=test_dir,
        region_save_to=partition_dir,
    )
    return tester
