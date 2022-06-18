import torch
from torchvision import transforms as T
from torch.cuda.amp import autocast
from simulation.agent.models.object_detector.model import Yolov1

class ObjectDetectorNetwork():
    def __init__(self, img_size=448, device="auto", iou_thresh=0.3, object_thresh=0.6):

        self.img_size = img_size
        self.stability_coeff = 1e-6
        self.device = torch.device(device) if device!="auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iou_thresh = iou_thresh
        self.object_thresh = object_thresh
        self.model = Yolov1(in_channels=3, split_size=7, num_boxes=2, num_classes=8).to(self.device)
        self.model.load_state_dict(torch.load("/simulation/agent/models/weights/yolo")["state_dict"])
        self.model = torch.jit.script(self.model.eval())
        self.transform = T.Compose([
            T.Resize((img_size, img_size))
        ])
        self.decoder = {
            0:"car_front",
            1:"car_back",
            2:"pedestrian",
            3:"stop",
            4:"go",
            5:"30",
            6:"60",
            7:"90"
        }
        self.traced_convert_cellboxes = torch.jit.trace(self.convert_cellboxes, torch.ones((1,882)))
    

    def convert_to_boxes(self, out):
        S: int = 7
        converted_pred = self.traced_convert_cellboxes(out).reshape(out.shape[0], S * S, -1) 
        converted_pred[..., 0] = converted_pred[..., 0].long()
        all_bboxes = []
        for ex_idx in range(out.shape[0]):
            bboxes = []
            for bbox_idx in range(S * S):
                bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
            all_bboxes.append(bboxes)
        return all_bboxes


    def convert_cellboxes(self, preds: torch.Tensor):
        S: int = 7
        preds = preds.to("cpu")
        batch_size = preds.shape[0]
        preds = preds.reshape(batch_size, 7, 7, 18)
        bboxes1 = preds[..., 9:13]
        bboxes2 = preds[..., 14:18]
        scores = torch.cat(
            (preds[..., 8].unsqueeze(0), preds[..., 13].unsqueeze(0)), dim=0
        )
        best_box = scores.argmax(0).unsqueeze(-1)
        best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
        cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
        x = 1 / S * (best_boxes[..., :1] + cell_indices)
        y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
        w_y = 1 / S * best_boxes[..., 2:4]
        converted_bboxes = torch.cat((x, y, w_y), dim=-1)
        predicted_class = preds[..., :8].argmax(-1).unsqueeze(-1)
        best_confidence = torch.max(preds[..., 8], preds[..., 13]).unsqueeze(
            -1
        )
        converted_preds = torch.cat(
            (predicted_class, best_confidence, converted_bboxes), dim=-1
        )
        return converted_preds


    def nms(self, bboxes, iou_threshold, obj_threshold):
        bboxes = [box for box in bboxes if box[1] > obj_threshold]
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
        processed_boxes = []
        while bboxes:
            cnt_box = bboxes.pop(0)
            bboxes = [box for box in bboxes if box[0] != cnt_box[0] or self.iou(torch.tensor(cnt_box[2:]),torch.tensor(box[2:])) <iou_threshold]
            processed_boxes.append(cnt_box)
        return processed_boxes


    def iou(self, preds, labels):
        box1_x1 = preds[..., 0:1] - preds[..., 2:3] / 2
        box1_y1 = preds[..., 1:2] - preds[..., 3:4] / 2
        box1_x2 = preds[..., 0:1] + preds[..., 2:3] / 2
        box1_y2 = preds[..., 1:2] + preds[..., 3:4] / 2
        box2_x1 = labels[..., 0:1] - labels[..., 2:3] / 2
        box2_y1 = labels[..., 1:2] - labels[..., 3:4] / 2
        box2_x2 = labels[..., 0:1] + labels[..., 2:3] / 2
        box2_y2 = labels[..., 1:2] + labels[..., 3:4] / 2
        x1 = torch.max(box1_x1, box2_x1)
        x2 = torch.min(box1_x2, box2_x2)
        y1 = torch.max(box1_y1, box2_y1)
        y2 = torch.min(box1_y2, box2_y2)
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
        return inter / (box1_area + box2_area - inter + self.stability_coeff)

    @autocast()
    @torch.no_grad()
    def __call__(self, image):
        image = self.transform(image)    
        boxes = self.model(image)
        boxes = self.convert_to_boxes(boxes)
        boxes=  self.nms(boxes[0], iou_threshold=self.iou_thresh, obj_threshold=self.object_thresh)
        ret = []
        for box in boxes:
            cnt = {}
            cnt["class"] = self.decoder[box[0]]
            cnt["confidence"] = box[1]
            cnt["boxes"] = box[2:]
            if cnt["class"] != "stop" and cnt["class"] != "go":
                if cnt["confidence"] < 0.7:
                    continue
            ret.append(cnt)
        return ret

