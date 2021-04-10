# # # detect lines
# # gray = cv2.cvtColor(np.array(np_im_org), cv2.COLOR_BGR2GRAY)
# # gray_blur = cv2.blur(gray, (5, 5))
# # edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)
# tmp = (out[0][0].detach().cpu().numpy()*255).astype(np.uint8)
# edges = cv2.Canny(tmp, 30, 60, apertureSize=3)
# plt.imshow(edges)
# lines = cv2.HoughLinesP(edges, rho=0.7, theta=np.pi / 180, threshold=70, minLineLength=20, maxLineGap=100)
# line_img = np.zeros_like(np_im_org)
# if lines is not None:
#     lines = lines[:, 0, :]
#     for x1, y1, x2, y2 in lines:
#         cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
# line_img = line_img[:, :, 0]


# model_in_line_img = torch.from_numpy(line_img / 255.).to(torch.float32).unsqueeze(0)
# model_in = torch.cat([model_in_img, torch.zeros((1,608,608))], dim=0)

# asd
# im = out
# im = torch.round(out)

# im = torch.sigmoid(out[0])
# im = torch.round(im)
# im[im >= 0.1] = 1
# im[im < 1] = 0
# plt.imshow(im[0][0].detach().cpu().numpy())
# plt.imshow(np_im_org)
# plt.figure()
# plt.imshow(out[0][0].detach().cpu().numpy())
# plt.figure()
# plt.imshow(np_im_org)
# plt.imshow(im[0][0].detach().cpu().numpy(), alpha=0.5)

# tmp = (out[0][0].detach().cpu().numpy()*255).astype(np.uint8)
# ret2, th2 = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plt.figure()
# plt.imshow(np_im_org)
# plt.imshow(th2, alpha=0.5)

# opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, np.ones((21,21),np.uint8))
# erosion = cv2.erode(opening, np.ones((7, 7),np.uint8), iterations=1)

# plt.figure()
# plt.imshow(np_im_org)
# plt.imshow(opening, alpha=0.5)
# plt.figure()
# plt.imshow(np_im_org)
# plt.imshow(erosion, alpha=0.5)

# final_prd = (im[0][0].detach().cpu().numpy()*255).astype(np.uint8)
# final_pred = opening
# final_pred = erosion


# # detect lines
# gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
# gray_blur = cv2.blur(gray, (5, 5))
# edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)
# lines = cv2.HoughLinesP(edges, rho=0.7, theta=np.pi / 180, threshold=70, minLineLength=20, maxLineGap=100)
# line_img = np.zeros_like(img)
# if lines is not None:
#     lines = lines[:, 0, :]
#     for x1, y1, x2, y2 in lines:
#         cv2.line(line_img, (x1, y1), (x2, y2), (255,255,255), 2)
# line_img = line_img[:, :, 0]