from .scene import ScenePCD, SceneMap, SceneTrav


class SceneBuilding():
    pcd = ScenePCD()
    # pcd.file_name = 'building2_9.pcd'
    # pcd.file_name = 'building_2F_4R.pcd'
    # pcd.file_name = 'building_LEE.pcd'
    # pcd.file_name = 'building_LEE_1F.pcd'
    # pcd.file_name = 'ETH_HPH.pcd'
    pcd.file_name = 'experiments/2F_2*1.pcd'

    map = SceneMap()
    map.resolution = 0.1
    map.ground_h = 0.0
    map.slice_dh = 1.0

    trav = SceneTrav()
    trav.kernel_size = 7
    trav.interval_min = 0.50
    trav.interval_free = 0.65
    trav.slope_max = 0.2
    trav.step_max = 0.3      # This factor influnce largely on the traversibility of the stairs/ slopes
    trav.standable_ratio = 0.10
    trav.cost_barrier = 50.0
    trav.safe_margin = 0.5
    trav.inflation = 0.2

