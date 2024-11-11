# WORKSPACE

# Load Bazel's http_archive rule
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_oci",
    sha256 = "e96d70faa4bace3e09fdb1d7d1441b838920f491588889ff9a7e2615afca5799",
    strip_prefix = "rules_oci-2.0.0-alpha2",
    url = "https://github.com/bazel-contrib/rules_oci/releases/download/v2.0.0-alpha2/rules_oci-v2.0.0-alpha2.tar.gz",
)

# Load rules_docker dependencies
load("@io_bazel_rules_docker//repositories:repositories.bzl", "rules_docker_dependencies", "rules_docker_toolchains")
rules_docker_dependencies()

rules_docker_toolchains()

# Enable container feature flags if needed
load("@io_bazel_rules_docker//container:container.bzl", "container_repositories")
container_repositories()
