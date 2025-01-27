
import os
import tempfile
import unittest
from pathlib import Path
from typing import List

from dpti.workflows.service.file_handler import LocalFileHandler

class TestLocalFileHandler(unittest.TestCase):
    def setUp(self):
        # 创建一个临时目录作为测试环境
        self.test_dir = tempfile.TemporaryDirectory()
        self.flow_trigger_dir = os.path.join(self.test_dir.name, "flow_trigger")
        self.flow_running_dir = os.path.join(self.test_dir.name, "flow_running")
        os.makedirs(self.flow_trigger_dir, exist_ok=True)
        os.makedirs(self.flow_running_dir, exist_ok=True)
        self.file_handler = LocalFileHandler(self.flow_trigger_dir, self.flow_running_dir)

    def tearDown(self):
        # 清理临时目录
        self.test_dir.cleanup()

    def test_initialization(self):
        self.assertEqual(self.file_handler.flow_trigger_dir, self.flow_trigger_dir)
        self.assertEqual(self.file_handler.flow_running_dir, self.flow_running_dir)
        self.assertEqual(self.file_handler.job_dirname, "default_job/")
        self.assertEqual(self.file_handler.job_dir, self.flow_running_dir)

    def test_use_job_info(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        self.assertEqual(self.file_handler.job_dirname, job_dirname)
        self.assertEqual(self.file_handler.job_dir, os.path.join(self.flow_running_dir, job_dirname))

    def test_create_job_dir(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        created_dir = self.file_handler.create_job_dir()
        self.assertTrue(os.path.isdir(created_dir))
        self.assertEqual(created_dir, self.file_handler.job_dir)

    def test_write_pure_file(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        file_path = "test_file.txt"
        file_content = "Hello, World!"
        abs_file_path = self.file_handler.write_pure_file(file_path, file_content)
        self.assertTrue(os.path.isfile(abs_file_path))
        with open(abs_file_path, 'r') as f:
            self.assertEqual(f.read(), file_content)
        self.assertIn(abs_file_path, self.file_handler.current_produced_paths)

    def test_upload_files(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        self.file_handler.create_job_dir()

        # 创建一个测试文件
        test_file_path = os.path.join(self.flow_trigger_dir, "test_file.txt")
        with open(test_file_path, 'w') as f:
            f.write("Test content")

        # 上传文件
        file_paths = ["test_file.txt"]
        produced_symlinks = self.file_handler.upload_files(file_paths, self.flow_trigger_dir)
        self.assertEqual(len(produced_symlinks), 1)
        self.assertTrue(os.path.islink(produced_symlinks[0]))
        self.assertIn(produced_symlinks[0], self.file_handler.current_produced_paths)

    def test_subdir_context(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        self.file_handler.create_job_dir()

        subdirname = "subdir"
        with self.file_handler.subdir_context(subdirname):
            self.assertEqual(self.file_handler.job_dir, os.path.join(self.flow_running_dir, job_dirname, subdirname))
        self.assertEqual(self.file_handler.job_dir, os.path.join(self.flow_running_dir, job_dirname))

    def test_create_relative_symlink_file(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        self.file_handler.create_job_dir()

        # 创建一个测试文件
        test_file_path = os.path.join(self.flow_trigger_dir, "test_file.txt")
        with open(test_file_path, 'w') as f:
            f.write("Test content")

        # 创建符号链接
        target_linkfile_path = self.file_handler.create_relative_symlink_file(
            file_path="test_file.txt",
            target_dir=self.file_handler.job_dir,
            work_base_dir=self.flow_trigger_dir
        )
        self.assertTrue(os.path.islink(target_linkfile_path))
        self.assertEqual(os.path.basename(target_linkfile_path), "test_file.txt")

    def test_ensure_create_job_dir(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        
        self.assertFalse(os.path.isdir(self.file_handler.job_dir))
        self.file_handler.write_pure_file("test_file.txt", "Hello, World!")
        self.assertTrue(os.path.isdir(self.file_handler.job_dir))
        
        self.file_handler.current_produced_paths = []  # 重置当前产生的路径
        self.file_handler.write_pure_file("test_file2.txt", "Hello again!")
        self.assertEqual(len(self.file_handler.current_produced_paths), 1)  #

    def test_link_files(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        self.file_handler.create_job_dir()

        test_file_path = os.path.join(self.flow_trigger_dir, "test_file.txt")
        with open(test_file_path, 'w') as f:
            f.write("Test content")

        produced_symlinks = self.file_handler.link_files(["test_file.txt"], self.flow_trigger_dir)
        self.assertEqual(len(produced_symlinks), 1)
        self.assertTrue(os.path.islink(produced_symlinks[0]))
        self.assertEqual(os.path.basename(produced_symlinks[0]), "test_file.txt")

        link_target = os.readlink(produced_symlinks[0])
        self.assertEqual(link_target, os.path.relpath(test_file_path, start=self.file_handler.job_dir))

        with self.assertRaises(RuntimeError):
            self.file_handler.link_files(["non_existent_file.txt"], self.flow_trigger_dir)
    def test_subdir_context_with_exception(self):
        job_dirname = "test_job"
        self.file_handler.use_job_info(job_dirname)
        self.file_handler.create_job_dir()

        subdirname = "subdir"
        original_job_dir = self.file_handler.job_dir

        with self.assertRaises(Exception):
            with self.file_handler.subdir_context(subdirname):
                self.assertEqual(self.file_handler.job_dir, os.path.join(original_job_dir, subdirname))
                raise Exception("Test exception")
        
        self.assertEqual(self.file_handler.job_dir, original_job_dir)

if __name__ == "__main__":
    unittest.main()