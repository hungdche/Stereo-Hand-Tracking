# Stereo-Hand-Tracking

## Main Idea

Traditional classical CV are good enough for detection fingers and index, as well as hand poses, but nost so well when occlusion occurs. We propose a method of estimating 3D hand poses, target especially for occlusion. 

### Inspiration

Hololens use ray point for hand interaction -> not intuitive. Stick to the basic approach of near-distance hand interaction -> needs hand detection for ZED, which places in front of the HMD. 

## Features (TODOs)

1. [ ] **Input Feeder:** Can run dataset, Zed Mini, and Realsense (if have time). If both dataset and Zed are provided, prioritize Zed

2. [ ] **Stereo Matching** perform stereo matching, can be skipped if using neural network, will have to dig deeper. 

3. [ ] **Classical Finger Tracking** using OpenCV. Tons of tutorial online

4. [ ] **Neural Network** train to detect hand poses during occlusion. If all 5 fingers can be detected, switch back to classical might be a better choice 

5. [ ] **Hand pose Simulation** need visualization in 3D to show that we can detect occlusion, consider using Pangolin 

## Credits 

This is CS498 Machine Perception final project by Henry Che @hungdche and Jeffrey Liu @jebbly. 