import os
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import numpy as np
import wandb
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips

# TensorBoard（用 prepare_output_and_logger 里创建的 tb_writer，不在全局新建）
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import time


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, args_dict):
    """
    训练入口（RAIN-GS + 支配集骨架初始化兼容版）
    必须同时把 training_report 的签名改为：
      def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                          testing_iterations, scene: Scene, renderFunc, renderArgs, run_dir):
        ...  # 日志落在 run_dir
    """
    # --------- 准备路径/日志 ----------
    first_iter = 0
    args = Namespace(**args_dict)
    tb_writer = prepare_output_and_logger(args, args.output_path, args.exp_name, args.project_name)
    run_dir = args.model_path  # 统一的落盘目录

    # 关键：把 model_path 同步到 dataset（以及 source_path 也同步一下更稳妥）
    if getattr(dataset, "model_path", None) in (None, ""):
        dataset.model_path = args.model_path
    if getattr(dataset, "source_path", None) in (None, ""):
        dataset.source_path = args.source_path

    # --------- 构建模型与场景 ----------
    divide_ratio = 0.7 if (args_dict.get('ours') or args_dict.get('ours_new')) else 0.8
    print(f"[Init] divide_ratio = {divide_ratio}")
    gaussians = GaussianModel(dataset.sh_degree, divide_ratio)
    scene = Scene(dataset, gaussians, args_dict=args_dict)

    # 如需扁平初始化，这里透传（若无对应参数也无碍）
    if hasattr(args, "flat_init"):
        gaussians.flat_init = args.flat_init
        gaussians.flat_t_mult = getattr(args, "flat_t_mult", 1.0)
        gaussians.flat_n_mult = getattr(args, "flat_n_mult", 0.2)

    # --------- 训练设置 / checkpoint 恢复 ----------
    gaussians.training_setup(opt)

    # RAIN 风格：推迟密度化窗口（暖启动）
    if args_dict.get("warmup_iter", 0) > 0:
        warm = int(args_dict["warmup_iter"])
        opt.densify_from_iter = max(getattr(opt, "densify_from_iter", 0), warm)
        opt.densify_until_iter = int(getattr(opt, "densify_until_iter", 0)) + warm
        print(f"[Warmup] densify_from_iter >= {opt.densify_from_iter}, densify_until_iter = {opt.densify_until_iter}")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # --------- 背景/计时 ----------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1

    # --------- 主循环 ----------
    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.cur_iter = iteration

        # GUI 交互
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, 0.0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        # 学习率调度（ours_new：暖启动偏移）
        if args_dict.get('ours_new'):
            if iteration >= args_dict.get("warmup_iter", 0):
                gaussians.update_learning_rate(iteration - args_dict["warmup_iter"])
        else:
            gaussians.update_learning_rate(iteration)

        # SH 阶提升
        if (args_dict.get('ours') or args_dict.get('ours_new')):
            if iteration >= 5000 and iteration % 1000 == 0:
                gaussians.oneupSHdegree()
        else:
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

        # 采样视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 调试
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # # c2f 低通（默认不低通，只有 --c2f 时启用）
        # c2f = args_dict.get('c2f', False)
        # low_pass = 0.0
        # if c2f:
        #     if iteration == 1 or (iteration % args_dict['c2f_every_step'] == 0 and iteration < opt.densify_until_iter):
        #         H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        #         N = gaussians.get_xyz.shape[0]
        #         low_pass = max(H * W / N / (9 * np.pi), 0.3)
        #         if args_dict['c2f_max_lowpass'] > 0:
        #             low_pass = min(low_pass, args_dict['c2f_max_lowpass'])
        #         print(f"[ITER {iteration}] Low pass filter : {low_pass:.3f}")

        # c2f（RAIN-GS 风格：仅随迭代渐进）
        c2f = args_dict.get('c2f', False)
        low_pass = 0.0
        if c2f:
            # 可从 args_dict 取自定义超参；没有就用默认
            lp_start   = float(args_dict.get('c2f_start', 1.5))
            lp_end     = float(args_dict.get('c2f_end',   0.3))
            lp_warmup  = int(args_dict.get('c2f_warmup', 0))
            lp_iters   = int(args_dict.get('c2f_iters',  20000))  # 渐进区间长度

            if iteration < lp_warmup:
                low_pass = lp_start
            else:
                t = min(1.0, max(0.0, (iteration - lp_warmup) / max(1, lp_iters)))
                low_pass = (1.0 - t) * lp_start + t * lp_end

            # 如需保持你原来的打印
            if iteration == 1 or iteration % 1000 == 0:
                print(f"[ITER {iteration}] Low pass (schedule): {low_pass:.3f}")



        # 前向
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, low_pass=low_pass)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg["depth"]

        # 损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        # --------- 无梯度区：日志/评估/保存/密度化/优化器/ckpt ----------
        with torch.no_grad():
            # 控制台进度
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "num_gaussians": f"{gaussians.get_xyz.shape[0]}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 评估 & 可视化
            elapsed_ms = iter_start.elapsed_time(iter_end)  # ms
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_ms,
                            testing_iterations, scene, render, (pipe, background), run_dir)

            # —— 额外：时间与显存写 TensorBoard —— 
            if tb_writer:
                tb_writer.add_scalar("time/iter_ms_cuda_event", elapsed_ms, iteration)
                if torch.cuda.is_available():
                    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    tb_writer.add_scalar("gpu/max_mem_MB", mem_mb, iteration)
                    torch.cuda.reset_peak_memory_stats()

            # 保存（非 ckpt）
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # 密度化/裁剪/不透明重置（在统一原子接口里完成）
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # —— densify 触发：一次性 densify+prune（原子） —— 
                stats = None
                allow_split = (iteration > getattr(opt, "densify_from_iter", 0))
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                        size_threshold, N=2, abe_split=False
                    )

                # opacity reset
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步进
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # checkpoint
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + f"/chkpnt{iteration}.pth")

            # 非 densify 步也记录一次 N（降低频率）
            if tb_writer and (iteration % 1000 == 0):
                tb_writer.add_scalar("gaussians/N", gaussians.get_xyz.shape[0], iteration)

    # 训练结束
    if tb_writer:
        tb_writer.close()



def prepare_output_and_logger(args, output_path, exp_name, project_name):
    if (not args.model_path) and (not exp_name):
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    elif (not args.model_path) and exp_name:
        args.model_path = os.path.join("./output", exp_name)

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, 'command_line.txt'), 'w') as file:
        file.write(' '.join(sys.argv))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        print("Logging progress to Tensorboard at {}".format(args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs, run_dir):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0; psnr_test = 0.0; lpips_test = 0.0; ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render",
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image)
                n = len(config['cameras'])
                psnr_test /= n; l1_test /= n; lpips_test /= n; ssim_test /= n

                print(f"\n[ITER {iteration}] Evaluating {config['name']}: "
                      f"L1 {l1_test} PSNR {psnr_test} LPIPS(vgg) {lpips_test} SSIM {ssim_test}")

                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, 'log_file.txt'), 'a') as f:
                    f.write(f"[ITER {iteration}] Evaluating {config['name']}: "
                            f"L1 {l1_test} PSNR {psnr_test} LPIPS(vgg) {lpips_test} SSIM {ssim_test}\n")

                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - lpips", lpips_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - ssim", ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    parser.add_argument("--init_pc", type=str, default=None,
                        help="Path to initial point cloud (.ply or COLMAP points3D.bin)")
    parser.add_argument("--init_pc_format", type=str, default="auto",
                        choices=["auto", "ply", "colmap_bin"],
                        help="Format of initial point cloud")
    # 可选：初始尺度 & 法向优先（RAIN-GS 的低通起步用得上）
    parser.add_argument("--init_sigma_t", type=float, default=1e-3, help="Init tangent sigma")
    parser.add_argument("--init_sigma_n", type=float, default=1e-3, help="Init normal/depth sigma")
    parser.add_argument("--init_opacity", type=float, default=0.1, help="Init opacity")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default='./output/')
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="gaussian-splatting")
    parser.add_argument("--c2f", action="store_true", default=False)
    # parser.add_argument("--c2f_every_step", type=float, default=1000, help="Recompute low pass filter size for every c2f_every_step iterations")
    # parser.add_argument("--c2f_max_lowpass", type=float, default=300, help="Maximum low pass filter size")
    parser.add_argument("--num_gaussians", type=int, default=1000000, help="Number of random initial gaussians to start with (default=1M for random)")
    parser.add_argument('--paper_random', action='store_true', help="Use the initialisation from the paper")
    parser.add_argument("--ours", action="store_true", help="Use our initialisation")
    parser.add_argument("--ours_new", action="store_true", help="Use our initialisation version 2")
    parser.add_argument("--warmup_iter", type=int, default=0)
    parser.add_argument("--train_from", type=str, default="random", choices=["random", "reprojection", "cluster", "noisy_sfm"])
    
    parser.add_argument("--c2f_start", type=float, default=1.5) # 最开始的低通强度
    parser.add_argument("--c2f_end",   type=float, default=0.3) # 结束时的低通强度
    parser.add_argument("--c2f_warmup", type=int,   default=0)
    parser.add_argument("--c2f_iters",  type=int,   default=20000) #从 c2f_start 平滑过渡到 c2f_end 所用的迭代长度（不含 warmup）。越大表示过渡越慢、低通放开得越晚。


    # 扁平初始化开关（与你的 gaussian_model.py 对齐）
    parser.add_argument("--flat_init", action="store_true", help="Use flattened Gaussian init (σz < σx=σy).")
    parser.add_argument("--flat_t_mult", type=float, default=1.0, help="Tangential multiplier for σx,σy at init.")
    parser.add_argument("--flat_n_mult", type=float, default=0.2, help="Normal multiplier for σz at init.")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.white_background = args.white_bg
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    args.eval = True
    outdoor_scenes = ['bicycle', 'flowers', 'garden', 'stump', 'treehill']
    indoor_scenes = ['room', 'counter', 'kitchen', 'bonsai']
    for scene_name in outdoor_scenes:
        if scene_name in args.source_path:
            args.images = "images_4"
            print("Using images_4 for outdoor scenes")
    for scene_name in indoor_scenes:
        if scene_name in args.source_path:
            args.images = "images_2"
            print("Using images_2 for indoor scenes")

    if args.ours or args.ours_new:
        print("========= USING OUR METHOD =========")
        args.c2f = True
        args.c2f_every_step = 1000
        args.c2f_max_lowpass = 300
    if args.ours_new:
        args.warmup_iter = 10000

    if args.ours and (args.train_from != "random"):
        parser.error("Our initialization version 1 can only be used with --train_from random")

    print(f"args: {args}")

    # GUI init（端口被占用则自增重试）
    while True:
        try:
            network_gui.init(args.ip, args.port)
            print(f"GUI server started at {args.ip}:{args.port}")
            break
        except Exception as e:
            args.port = args.port + 1
            print(f"Failed to start GUI server, retrying with port {args.port}...")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, args.__dict__)

    print("\nTraining complete.")
