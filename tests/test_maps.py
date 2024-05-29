import os
from torchdrivesim.map import find_map_config, find_wrong_way_stoplines


class TestSavedMaps:
    def test_wrong_way_stoplines(self):
        wrong_way_stoplines = {map_name: find_wrong_way_stoplines(find_map_config(map_name))
                               for map_name in sorted(os.listdir('torchdrivesim/resources/maps'))}
        for map_name, stopline_ids in wrong_way_stoplines.items():
            if stopline_ids:
                print(f'{map_name}: {stopline_ids}')
        assert all(not stopline_ids for stopline_ids in wrong_way_stoplines.values())
        