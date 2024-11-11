# WORKSPACE

# Load Bazel's http_archive rule
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "b1e80761a8a8243d03ebca8845e9cc1ba6c82ce7c5179ce2b295cd36f7e394bf",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.25.0/rules_docker-v0.25.0.tar.gz"],
)

# Load rules_docker dependencies
load("@io_bazel_rules_docker//repositories:repositories.bzl", "rules_docker_dependencies", "rules_docker_toolchains")
rules_docker_dependencies()

rules_docker_toolchains()

# Enable container feature flags if needed
load("@io_bazel_rules_docker//container:container.bzl", "container_repositories")
container_repositories()
