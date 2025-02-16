# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# To run the example, start two terminals:
# > python pir_setup.py --in_path examples/data/pir_server_data.csv --key_columns id --label_columns label \
# > --count_per_query 1 -max_label_length 256 \
# > --oprf_key_path oprf_key.bin --setup_path setup_path

import time

from absl import app, flags

import spu.libspu.link as link
import spu.libspu.logging as logging
import spu.pir as pir

flags.DEFINE_integer("rank", 0, "rank: 0/1/2...")
flags.DEFINE_string("party_ips", "127.0.0.1:61307,127.0.0.1:61308", "party addresses")

flags.DEFINE_string("in_path", "data.csv", "data input path")
flags.DEFINE_string("key_columns", "id", "csv file key filed name")
flags.DEFINE_string("label_columns", "label", "csv file label filed name")
flags.DEFINE_integer("count_per_query", 1, "count_per_query")
flags.DEFINE_integer("max_label_length", 256, "max_label_length")
flags.DEFINE_string("setup_path", "setup_path", "data output path")

flags.DEFINE_boolean("compressed", False, "compress seal he plaintext")
flags.DEFINE_integer("bucket_size", 1000000, "bucket size of pir query")
flags.DEFINE_integer(
    "max_items_per_bin", 0, "max items per bin, i.e. Interpolate polynomial max degree"
)

FLAGS = flags.FLAGS


def setup_link(rank):
    lctx_desc = link.Desc()
    lctx_desc.id = f"root"

    lctx_desc.recv_timeout_ms = 2 * 60 * 1000
    # lctx_desc.connect_retry_times = 180

    ips = FLAGS.party_ips.split(",")
    for i, ip in enumerate(ips):
        lctx_desc.add_party(f"id_{i}", ip)
        print(f"id_{i} = {ip}")

    return link.create_brpc(lctx_desc, rank)


def main(_):
    opts = logging.LogOptions()
    opts.system_log_path = "./tmp/spu.log"
    opts.enable_console_logger = True
    opts.log_level = logging.LogLevel.INFO
    logging.setup_logging(opts)

    key_columns = FLAGS.key_columns.split(",")
    label_columns = FLAGS.label_columns.split(",")

    link = setup_link(FLAGS.rank)

    start = time.time()

    config = pir.PirSetupConfig(
        pir_protocol=pir.PirProtocol.Value('KEYWORD_PIR_LABELED_PSI'),
        store_type=pir.KvStoreType.Value('LEVELDB_KV_STORE'),
        input_path=FLAGS.in_path,
        key_columns=key_columns,
        label_columns=label_columns,
        num_per_query=FLAGS.count_per_query,
        label_max_len=FLAGS.max_label_length,
        oprf_key_path="",
        setup_path='::memory',
        compressed=FLAGS.compressed,
        bucket_size=FLAGS.bucket_size,
        max_items_per_bin=FLAGS.max_items_per_bin,
    )

    report = pir.pir_memory_server(link, config)
    print(f"data_count: {report.data_count}")
    print(f"memory server cost time: {time.time() - start}")


if __name__ == '__main__':
    app.run(main)
