# WORKSPACE

# Load Bazel's http_archive rule
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Add bazel_skylib
http_archive(
    name = "bazel_skylib",
    sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz"],
)

# Update rules_docker to newer version
http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "59d5b42ac315e7b6cf8cfa242950aa3179f5a20288f7fb500f1f05d70751c428",
    strip_prefix = "rules_docker-0.17.0",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.17.0/rules_docker-v0.17.0.tar.gz"],
)

# Load Docker rules
load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)
container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")
container_deps()

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
)
