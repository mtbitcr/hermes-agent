# Owner app of the journeys stack: the released image plus the proxy's public CA certificate, as one file.
# Build folder: ca.pem only. APP_IMAGE_DIGEST is the digest reference of the released owner app image.
# No user, variable, label or command is added; the settings file points NODE_EXTRA_CA_CERTS at this file.
ARG APP_IMAGE_DIGEST
FROM ${APP_IMAGE_DIGEST}

# BuildKit gives a folder that COPY --chmod creates the same mode as the file (0644, which cannot be searched),
# so the file goes straight into /usr/local/share, which the image already has.
COPY --chmod=0644 ca.pem /usr/local/share/raphael-journeys-ca.pem
