import mmcv
import os
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch

import pandas as pd


cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../../../"))

from lib.vis_utils.colormap import colormap
from lib.utils.mask_utils import cocosegm2mask, get_edge
from core.utils.data_utils import crop_resize_by_warp_affine, read_image_mmcv
from core.gdrn_modeling.datasets.dataset_factory import register_datasets
from lib.pysixd import misc
from transforms3d.quaternions import quat2mat
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer

from core.utils.my_visualizer import MyVisualizer, _GREY, _GREEN, _BLUE , _RED

import trimesh


def load_predicted_csv(fname):
    df = pd.read_csv(fname)
    info_list = df.to_dict("records")
    return info_list

def parse_Rt_in_csv(_item):
    return np.array([float(i) for i in _item.strip(" ").split(" ")])


out_size = 1000
score_thr = 0.3
colors = colormap(rgb=False, maximum=255)

id2obj = {
    1: "ape",
    #  2: 'benchvise',
    #  3: 'bowl',
    #  4: 'camera',
    5: "can",
    6: "cat",
    #  7: 'cup',
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    #  13: 'iron',
    #  14: 'lamp',
    #  15: 'phone'
}
objects = list(id2obj.values())

tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
image_tensor = torch.empty((out_size, out_size, 4), **tensor_kwargs).detach()
seg_tensor = torch.empty((out_size, out_size, 4), **tensor_kwargs).detach()
# image_tensor = torch.empty((480, 640, 4), **tensor_kwargs).detach()

model_dir = "datasets/BOP_DATASETS/lmo/models/"

model_paths = [osp.join(model_dir, f"obj_{obj_id:06d}.ply") for obj_id in id2obj]

ren = EGLRenderer(
    model_paths,
    vertex_scale=0.001,
    use_cache=True,
    width=out_size,
    height=out_size,
)

# NOTE:
# pred_path = "output/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e/inference_model_final/lmo_test/a6-cPnP-AugAAETrunc-BG0.5-lmo-real-pbr0.1-40e-test_lmo_test_preds.pkl"
# pred_path = "./output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/inference_model_final_wo_optim/lmo_bop_test/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-lmo-test-iter0_lmo-test.csv"
pred_path = "./output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/inference_model_final/lmo_bop_test/convnext-a6-AugCosyAAEGray-BG05-mlL1-DMask-amodalClipBox-classAware-lmo-test-iter0_lmo-test.csv"
# vis_dir = "output/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e/inference_model_final/lmo_test/vis_gt_pred"
vis_dir = "output/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e/inference_model_final/lmo_test/lmo_vis_gt_pred_full"

mmcv.mkdir_or_exist(vis_dir)

preds_csv = load_predicted_csv(pred_path)


preds = {}
for item in preds_csv:
    scene_id = item['scene_id']
    im_id = item['im_id']
    obj_id = item['obj_id']
    obj_name = id2obj[obj_id] 
    file_name = f"{scene_id}/{im_id}"  
    print('file_name: ', file_name)
    if obj_name not in preds:
        preds[obj_name] = {}

    if file_name not in preds[obj_name]:
        preds[obj_name][file_name] = {}

    preds[obj_name][file_name]["score"] = item["score"]
    preds[obj_name][file_name]["R"] = parse_Rt_in_csv(item["R"]).reshape(3, 3)
    preds[obj_name][file_name]["t"] = parse_Rt_in_csv(item["t"])


# dataset_name = "lmo_test"
dataset_name = "lmo_bop_test"
# dataset_name = "lmo_bop_test"

print(dataset_name)
register_datasets([dataset_name])

meta = MetadataCatalog.get(dataset_name)
print("MetadataCatalog: ", meta)
objs = meta.objs

dset_dicts = DatasetCatalog.get(dataset_name)
for d in tqdm(dset_dicts):
    K = d["cam"]
    file_name = d["file_name"]
    path_parts = file_name.split('/')
    scene_id = path_parts[-3] 
    scene_id = str(int(scene_id))
    im_id_with_extension = path_parts[-1]  
    im_id = os.path.splitext(im_id_with_extension)[0]  
    im_id = str(int(im_id))  
    converted_format = f"{scene_id}/{im_id}"
    
    # ====== 讀圖後：避免任何固定尺寸畫布與硬翻轉 ======
    img = read_image_mmcv(file_name, format="BGR") # 保持原始 640x480
    # 若你的程式裡有手動翻轉（常見像 img = img[::-1, ::-1] 或 cv2.rotate(...,180)）
    # 直接移除那行；真的需要時才加下列這個「可選」開關
    FORCE_NO_FLIP = True
    if not FORCE_NO_FLIP:
        pass  # 這裡留白即可（預設不翻轉）
    
    scene_im_id_split = d["scene_im_id"].split("/")
    scene_id = scene_im_id_split[0]
    im_id = int(scene_im_id_split[1])

    imH, imW = img.shape[:2]
    H, W = img.shape[:2]
    
    annos = d["annotations"]
    masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
    bboxes = [anno["bbox"] for anno in annos]
    bbox_modes = [anno["bbox_mode"] for anno in annos]
    bboxes_xyxy = np.array(
        [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
    )
    kpts_3d_list = [anno["bbox3d_and_center"] for anno in annos]
    
    quats = [anno["quat"] for anno in annos]
    transes = [anno["trans"] for anno in annos]
    Rs = [quat2mat(quat) for quat in quats]
    # 0-based label
    cat_ids = [anno["category_id"] for anno in annos]

    kpts_2d = [misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]

    obj_names = [objs[cat_id] for cat_id in cat_ids]

    kpts_2d_est = []
    est_Rs = []
    est_ts = []

    kpts_2d_gt = []
    gt_Rs = []
    gt_ts = []

    kpts_3d_list_sel = []
    labels = []

    maxx, maxy, minx, miny = 0, 0, 1000, 1000
    for anno_i, anno in enumerate(annos):
        kpt_2d_gt = kpts_2d[anno_i]
        obj_name = obj_names[anno_i]
        try:
            # R_est = preds[obj_name][file_name]["R"]
            # t_est = preds[obj_name][file_name]["t"]
            # score = preds[obj_name][file_name]["score"]
            R_est = preds[obj_name][converted_format]["R"]
            t_est = preds[obj_name][converted_format]["t"]
            score = preds[obj_name][converted_format]["score"]
        except:
            continue
        if score < score_thr:
            continue

        labels.append(objects.index(obj_name))  # 0-based label

        est_Rs.append(R_est)
        est_ts.append(t_est)

        kpts_3d_list_sel.append(kpts_3d_list[anno_i])
        kpt_2d_est = misc.project_pts(kpts_3d_list[anno_i], K, R_est, t_est)
        kpts_2d_est.append(kpt_2d_est)

        gt_Rs.append(Rs[anno_i])
        gt_ts.append(transes[anno_i])
        kpts_2d_gt.append(kpts_2d[anno_i])

        for i in range(len(kpt_2d_est)):
            maxx, maxy, minx, miny = (
                max(maxx, kpt_2d_est[i][0]),
                max(maxy, kpt_2d_est[i][1]),
                min(minx, kpt_2d_est[i][0]),
                min(miny, kpt_2d_est[i][1]),
            )
            maxx, maxy, minx, miny = (
                max(maxx, kpt_2d_gt[i][0]),
                max(maxy, kpt_2d_gt[i][1]),
                min(minx, kpt_2d_gt[i][0]),
                min(miny, kpt_2d_gt[i][1]),
            )
    center = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
    scale = max(maxx - minx, maxy - miny) * 1.5  # + 10
    crop_minx = max(0, center[0] - scale / 2)
    crop_miny = max(0, center[1] - scale / 2)
    crop_maxx = min(imW - 1, center[0] + scale / 2)
    crop_maxy = min(imH - 1, center[1] + scale / 2)
    scale = min(scale, min(crop_maxx - crop_minx, crop_maxy - crop_miny))

    zoomed_im = crop_resize_by_warp_affine(img, center, scale, out_size)
    im_zoom_gray = mmcv.bgr2gray(zoomed_im, keepdim=True)
    im_zoom_gray_3 = np.concatenate([im_zoom_gray, im_zoom_gray, im_zoom_gray], axis=2)
    # print(im_zoom_gray.shape)
    K_zoom = K.copy()
    K_zoom[0, 2] -= center[0] - scale / 2
    K_zoom[1, 2] -= center[1] - scale / 2
    K_zoom[0, :] *= out_size / scale
    K_zoom[1, :] *= out_size / scale

    gt_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(gt_Rs, gt_ts)]
    poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

    ren.render(
        labels,
        poses,
        K=K_zoom,
        image_tensor=image_tensor,
        background=im_zoom_gray_3,
    )
    ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

    # gt_masks = []
    # est_masks = []
    for label, gt_pose, est_pose in zip(labels, gt_poses, poses):
        ren.render([label], [gt_pose], K=K_zoom, seg_tensor=seg_tensor)
        gt_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

        ren.render([label], [est_pose], K=K_zoom, seg_tensor=seg_tensor)
        est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")

        gt_edge = get_edge(gt_mask, bw=3, out_channel=1)
        est_edge = get_edge(est_mask, bw=3, out_channel=1)

        # zoomed_im[gt_edge != 0] = np.array(mmcv.color_val("blue"))
        # zoomed_im[est_edge != 0] = np.array(mmcv.color_val("green"))

        ren_bgr[gt_edge != 0] = np.array(mmcv.color_val("blue"))
        ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))

    vis_im = ren_bgr

    # vis_im_add = (im_zoom_gray_3 * 0.3 + ren_bgr * 0.7).astype("uint8")

    kpts_2d_gt_zoom = [
        misc.project_pts(kpt3d, K_zoom, R, t) for kpt3d, R, t in zip(kpts_3d_list_sel, gt_Rs, gt_ts)]
    # print('kpts_2d_gt_zoom: ', kpts_2d_gt_zoom)
    kpts_2d_est_zoom = [
        misc.project_pts(kpt3d, K_zoom, R, t) for kpt3d, R, t in zip(kpts_3d_list_sel, est_Rs, est_ts)]
    # print('kpts_2d_est_zoom: ', kpts_2d_est_zoom)
   
    # 建一個 Visualizer 以 zoom 之後（無黑邊）的影像為底
    visualizer = MyVisualizer(zoomed_im[:, :, ::-1],meta)  # BGR->RGB 給 Visualizer
    
    # ====== 對每個物件畫 3D bbox 與 XYZ 三軸 ======
    # 你本來就會有 pose 與對應 model 的點雲/八個角資訊，下方示範如何取得投影與繪製
    # R: (3,3), t: (3,), K: (3,3) ;  model_bbox_3d: (8,3) 物件 3D bbox 的八個角點（模型座標）
    # center_3d: (1,3) 物件中心（模型座標）
    # 1) 畫 GT/Pred 的 3D bbox（8 角 + 中心，已內建畫法）
    linewidth = 3
    # for kpt2d_gt, kpt2d_est in zip(kpts_2d_gt_zoom, kpts_2d_est_zoom):
    #     visualizer.draw_bbox3d_and_center(
    #         kpt2d_gt, top_color=_BLUE, bottom_color=_GREY,
    #         linewidth=linewidth, draw_center=True
    #     )
    #     visualizer.draw_bbox3d_and_center(
    #         kpt2d_est, top_color=_GREEN, bottom_color=_GREY, 
    #         linewidth=linewidth, draw_center=True
    #     )
    # vis_im = visualizer.get_output()
    # 1) 專案已有計算 2D bbox 的工具（若需要 2D 框）
    # xyxy = misc.compute_2d_bbox_xyxy_from_pose(model_pts, np.hstack([R, t.reshape(3,1)]), K, width=W, height=H, clip=True)
    # x1,y1,x2,y2 = [int(round(v)) for v in xyxy]  # 可用來畫 2D 框（可選）

    # 2) 在物體中心畫 XYZ 三軸（用「Pred pose」）
    #    使用 3D bbox 的 8 個角與中心（kpts_3d_list_sel 內 9 點：前 8 角 + 最後一點為中心）
    for kpt3d, R_est, t_est in zip(kpts_3d_list_sel, est_Rs, est_ts):
        # 投影 3D bbox 八角 + 中心
        kpt2d = misc.project_pts(kpt3d, K_zoom, R_est, t_est)
        visualizer.draw_bbox3d_and_center(
            kpt2d,
            top_color=(0,1,0), bottom_color=(0,0,1),
            linewidth=2, draw_center=True
        )
        # 算三軸端點
        corners_3d = kpt3d[:8]
        center_3d  = kpt3d[-1]
        # 軸長：取 3D bbox 對角線的一小部分；避免太短或太長
        diag = np.linalg.norm(corners_3d.ptp(axis=0))
        axis_len = max(0.10 * diag, 0.05)  # 依模型單位可微調

        # 依「右手座標」定義出三軸端點
        X_end = center_3d + np.array([axis_len, 0, 0])
        Y_end = center_3d + np.array([0, axis_len, 0])
        Z_end = center_3d + np.array([0, 0, axis_len])

        # axes_3d = np.vstack([X_end, Y_end, Z_end, center_3d])  # 4 點
        # axes_2d = misc.project_pts(axes_3d, K_zoom, R_est, t_est)
        axis_pts3d = np.vstack([X_end, Y_end, Z_end, center_3d])
        axis_pts2d = misc.project_pts(axis_pts3d, K_zoom, R_est, t_est)

        # 顏色：X=藍、Y=綠、Z=紅（或依你喜好調換）
        # visualizer.draw_axis3d_and_center(
        #     keypoints=axes_2d.astype(np.float32),
        #     up_color    = _RED,    # Z
        #     right_color = _GREEN,  # Y
        #     front_color = _BLUE,   # X
        #     center_color= _GREY,
        #     linewidth   = 2,
        #     draw_points = False,
        #     draw_center = True,
        # )
        visualizer.draw_axis3d_and_center(
            keypoints=axis_pts2d.astype(np.float32),
            up_color=(1,0,0), right_color=(0,1,0), front_color=(0,0,1),
            center_color=(0.5,0.5,0.5),
            linewidth=2, draw_center=True
        )
    # 3) 產出影像並存檔（注意 RGB->BGR）
    # vis_im = visualizer.get_output()          # VisImage
    # save_path_im  = osp.join(vis_dir, f"{scene_id}_{im_id:06d}_im.png")
    # save_path_vis = osp.join(vis_dir, f"{scene_id}_{im_id:06d}_gt_est.png")

    # 原圖（BGR）與可視化（RGB->BGR）
    # mmcv.imwrite(zoomed_im, save_path_im)
    # mmcv.imwrite(vis_im.get_image()[:, :, ::-1], save_path_vis)
    
    out_path = osp.join(vis_dir, f"{scene_id}_{im_id:06d}_gt_est.png")
    mmcv.imwrite(visualizer.get_output().get_image()[:, :, ::-1], out_path)
    
    # 儲存結果
    # save_path = osp.join(vis_dir, "{}_{:06d}_gt_est.png".format(scene_id, im_id))
    # vis_im.save(save_path)

    # save_path_0 = osp.join(vis_dir, "{}_{:06d}_im.png".format(scene_id, im_id))
    # mmcv.imwrite(zoomed_im, save_path_0)

    # save_path = osp.join(vis_dir, "{}_{:06d}_gt_est.png".format(scene_id, im_id))
    # from detectron2.utils.visualizer import VisImage
    # import cv2
    # image = vis_im.get_image()
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('Image', image)  # Show the image in a window
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()  # Close all windows
    
    # vis_im_img = vis_im.get_image()
    #mmcv.imwrite(vis_im, save_path)
    # mmcv.imwrite(vis_im_img, save_path)

    # if True:
    #     # grid_show([zoomed_im[:, :, ::-1], vis_im[:, :, ::-1]], ["im", "est"], row=1, col=2)
    #     # im_show = cv2.hconcat([zoomed_im, vis_im, vis_im_add])
    #     im_show = cv2.hconcat([zoomed_im, vis_im])
    #     cv2.imshow("im_est", im_show)
    #     if cv2.waitKey(0) == 27:
    #         break  # esc to quit
