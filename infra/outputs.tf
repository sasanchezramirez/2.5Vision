output "server_public_ip" {
  description = "La dirección IP pública de nuestro servidor ML."
  value       = aws_instance.ml_server.public_ip
}

output "s3_bucket_name" {
  description = "El nombre de nuestro bucket S3 para los datos."
  value       = aws_s3_bucket.vision_data.id
}