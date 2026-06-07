# processor/import_processor/nodes/node_pdf_to_md.py
import json
import logging
import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.base import BaseNode, setup_logging
from processor.exceptions import FileProcessingError, StateFieldError, PdfConversionError
from processor.state import ImportGraphState


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        # 1.参数校验并返回Path结果
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)
        # # 2.上传PDF文件到MinerU并轮训结果解析
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        # # 3.将mineru中转化好的压缩文件进行下载并且解压出来
        md_path = self._step_3_download_and_extract(zip_url,output_dir_obj, pdf_path_obj.stem) # 第三个参数是去掉扩展名后的主文件名
        self.logger.info(f"PDF文件转换完成，保存路径：{md_path}")
        # 4.读取md文件中的内容
        with open(md_path,"r",encoding="utf-8") as f:
            md_content = f.read()
        # 5.更新state
        state["md_path"] = md_path
        state["md_content"] = md_content

        return state

    def _step_1_validate_paths(self, state):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：StateFieldError(参数缺失)、FileProcessingError(文件无效)
        """
        # 1. 判断上传的pdf路径文件是否存在，不存在直接返回报错
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(
                field_name = "pdf_path",
                expected_type = str
            )
        # 2. 判断文件夹是否存在
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(
                field_name = "file_dir",
                expected_type = str
            )
        # 3. 将字符串路径转换为Path路径
        pdf_path_obj = Path(pdf_path)
        file_path_obj = Path(file_dir)
        # 4. 判断pdf文件是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=fr"PDF文件{pdf_path_obj}不存在")

        # 5.确保输出目录存在，不存在则递归创建
        if not file_path_obj.exists():
            self.logger.info(f"输出目录{file_path_obj}不存在，自动创建{file_path_obj.absolute()}")
            file_path_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_path_obj

    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ConfigurationError(配置缺失)、PdfConversionError(请求/上传失败)、TimeoutError(任务超时)
        """

        token = self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm" # 模型的版本；多模态模型
        }
        # 1.从MinerU获取到上传的链接
        response = requests.post(url, headers=header, json=data)
        # 判断链接是否成功
        if response.status_code != 200:
            raise PdfConversionError(message=f"获取上传链接相应失败:状态码：{response.status_code},响应结果：{response}")

        result = response.json()
        # 校验业务码 ，只有是0的时候才算是业务上链接成功
        if result["code"] !=0:
            raise PdfConversionError(message=f"获取上传链接失败：返回数据：{result}")

        # 2. 从返回的结果result中获取到url和batchid
        # batchid：批量提取任务 id，可用于批量查询解析结果
        batch_id = result["data"]["batch_id"]
        url = result["data"]["file_urls"][0]

        # 3.根据url上传链接进行上传文件
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(url, data=f,timeout=60)
            self.logger.info(f"upload done: status={res_upload.status_code}, body={res_upload.text}")

            if res_upload.status_code != 200:
                raise PdfConversionError(message=f"上传文件失败:状态码：{res_upload.status_code},响应结果：{res_upload}")
            self.logger.info(f"上传文件成功:状态码：{res_upload.status_code},响应结果：{res_upload}")

        # 4.通过轮询 批量获取任务结果
        poll_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"

        start_time = time.time()# 记录开始时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 3  # 轮询间隔时间
        self.logger.info(f"【任务轮询】最大超时：{timeout_seconds}s，batch_id：{batch_id}")

        # 4. 根据batch_id轮询任务状态直到成功"done"
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise TimeoutError(f"【任务轮询】超时！任务处理超{timeout_seconds}秒，batch_id：{batch_id}")
            # 发起轮询请求，短超时10秒，异常则重试
            try:
                res_poll = requests.get(poll_url, headers=header,timeout=10)
            except Exception as e:
                self.logger.warning(f"【任务轮询】网络请求异常，{poll_interval}秒后重试：{str(e)}，bactch_id：{batch_id}")
                time.sleep(poll_interval)
                continue
            # 处理响应错误
            if res_poll.status_code != 200:
                raise PdfConversionError(f"【任务轮询】HTTP请求失败，状态码：{res_poll.status_code}，响应内容：{res_poll}")

            #解析轮询结果，校验业务状态
            poll_data = res_poll.json()
            if poll_data["code"] != 0:
                raise PdfConversionError(f"【任务轮询】业务错误，返回数据：{poll_data}")

            extract_results = poll_data["data"]["extract_result"]
            # 获取结果
            result_item = extract_results[0]
            data_state = result_item["state"]

            # 状态为done
            if data_state == "done":
                self.logger.info(f"【任务轮询】解析任务完成！总耗时{int(elapsed_time)}s，bactch_id：{batch_id}")

                full_zip_url = result_item["full_zip_url"]
                self.logger.info(f"【任务轮询】返回ZIP包下载链接：{full_zip_url}，bactch_id：{batch_id}")

                return full_zip_url
            elif data_state == "failed":
                err_msg = result_item.get("err_msg", "未知错误，无具体信息")
                raise PdfConversionError(f"【任务轮询】解析任务失败！batch_id：{batch_id}，错误信息：{err_msg}")

            else:
                self.logger.info(
                    f"【任务轮询】处理中... 已耗时{int(elapsed_time)}s，状态：{data_state}， batch_id：{batch_id}")
                time.sleep(poll_interval)

    def _step_3_download_and_extract(self, zip_url:str, output_dir_obj:Path, pdf_stem:str):
        # 第三个参数是去掉扩展名后的主文件名
        """
              步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
              核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
              参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
              返回：最终MD文件的字符串格式绝对路径
              异常：RuntimeError(下载失败)
        """
        # 1. 下载zip包
        self.log_step(f"【ZIP下载】开始下载ZIP包：{zip_url} ...")
        response = requests.get(zip_url)

        # 对相应结果进行校验
        if response.status_code != 200:
            raise RuntimeError(f"【ZIP下载】下载失败，状态码：{response.status_code}，响应内容：{response.text}")
        # 拼接zip包的路径并保存
        zip_save_path = output_dir_obj / f"{pdf_stem}.zip"
        with open (zip_save_path, "wb") as f: # 覆盖写
            f.write(response.content)
        self.logger.info(f"【ZIP下载】ZIP包下载成功：保存路径：{zip_save_path}")

        # 2. 清理旧目录并且解压
        # 先删除旧的目录，确保环境干净
        extract_target_dir = output_dir_obj / pdf_stem
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        self.logger.info(f"【ZIP解压】已清空旧的解压目录：{extract_target_dir}")

        # 创建解压目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 进行解压
        self.logger.info(f"【ZIP解压】开始解压ZIP包：{zip_save_path} ...")
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)
        self.logger.info(f"【ZIP解压】ZIP包解压完成：保存目录：{extract_target_dir}")

        # 4.将fill.md文档修改名字到pdf_stem.md
        self.logger.info(f"【MD重命名】找到MinerU生成的full.md文件")
        target_md_file = extract_target_dir / "full.md"
        self.logger.info(f"【MD重命名】开始将full.md文件进行重命名")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        target_md_file.rename(new_md_path)
        self.logger.info(f"【MD重命名】重命名成功，文件名：{pdf_stem}.md")
        # 5.进行返回
        return str(new_md_path.absolute())



if __name__ == "__main__":

    setup_logging()

    init_state = {
        "pdf_path": r"D:\\课程\\尚硅谷\\掌柜智库\\2.资料\\04-设备手册汇总\\doc\\hak180产品安全手册.pdf",
        "file_dir": r"D:\output"
    }
    node_pdf_to_md = NodePDFToMD()
    result = node_pdf_to_md(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))